"""Fixed-geometry control-amplitude sweep for multi-output oscillator switches.

Run from repository root: python analysis/control_nonlinearity.py --help
No optimiser or C++ changes. Zero control is achieved via the case-specific
amplitude, not a zero-amplitude base configuration.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import shutil
import traceback
import numpy as np

from simulation import load_configuration, run_simulation, save_json, fingerprint
from diagnose_switch import ROOT, phase_analyser
from switch_cycles import refine_timestep


def build_case(base, g, case, seed, divisor):
    drives = deepcopy(base.values['drives'])
    # Case-specific amplitudes; never alter geometries, periods or directions.
    drives[0]['amplitude'] *= int(case != 'control_only')
    drives[1]['amplitude'] *= g * int(case != 'input_only')
    adjusted = base.updated(drives=drives, seed=seed)
    return refine_timestep(adjusted, divisor)


def signal_metrics(analyser, bead, drive, window):
    m = analyser.measure_signal_locking(
        bead, drive['period'], window[0], window[1],
        signal_phase=drive['phase'])
    if m['status'] != 'ok':
        raise ValueError('Phase metric unavailable: ' + str(m['status']))
    return dict(R=float(m['phase_consistency_R']),
                mismatch=float(m['frequency_mismatch_cycles_per_time']),
                slips=int(m['slip_candidates_total']))


def run_job(base, g, case, rep, seed, divisor, window, exe, folder, keep):
    config = build_case(base, g, case, seed, divisor)
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder/'configuration.json', {'configuration':config.values})
    try:
        reader = run_simulation(config, exe, folder/'simulation')
        analyser = phase_analyser(config, reader)
        drive = config.values['drives'][0]
        outputs = config.targets('output')
        measurements = {'input':signal_metrics(analyser,config.targets('input')[0],drive,window)}
        for bead in outputs:
            measurements[f'output_{bead}'] = signal_metrics(analyser,bead,drive,window)
        result = {'g':g,'case':case,'repeat':rep,'seed':seed,'status':'ok',
                  'measurements':measurements}
        if not keep:
            shutil.rmtree(folder/'simulation')
    except Exception as exc:
        result = {'g':g,'case':case,'repeat':rep,'seed':seed,'status':'error',
                  'error':str(exc),'traceback':traceback.format_exc()}
    save_json(folder/'metrics.json', result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--g-min',type=float,default=0.)
    p.add_argument('--g-max',type=float,default=2.)
    p.add_argument('--points',type=int,default=9)
    p.add_argument('--runs',type=int,default=5)
    p.add_argument('--cores',type=int,default=3)
    p.add_argument('--seed',type=int,default=8300000)
    p.add_argument('--window',type=float,nargs=2,default=[60,95])
    p.add_argument('--dt-divisor',type=int,choices=[1,2,4],default=1)
    p.add_argument('--exe',type=Path,default=ROOT/'src/rowers')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--keep',action='store_true',help='Keep bulky raw trajectories')
    p.add_argument('--control-only',action='store_true',help='Run control-only comparison at every g')
    args=p.parse_args()
    base=load_configuration(args.config)
    if len(base.values['drives'])!=2 or len(base.targets('input'))!=1 or len(base.targets('control'))!=1 or not base.targets('output'):
        p.error('Need two drives, singleton input and control, and at least one output')
    if base.targets('input')[0] in base.targets('output') or base.targets('control')[0] in base.targets('output'):
        p.error('Output beads must differ from input and control')
    if not all(np.isfinite([args.g_min,args.g_max,*args.window])) or not 0<=args.g_min<args.g_max:
        p.error('Require finite 0 <= g-min < g-max')
    if args.points<2 or args.runs<1 or args.cores<1 or args.seed<0:
        p.error('points >= 2, runs/cores >= 1, seed >= 0 required')
    if not 0<=args.window[0]<args.window[1]<base.values['total_time']:
        p.error('Window must be strictly inside configuration total_time')
    if any(d['start']>args.window[0] or d['stop']<args.window[1] for d in base.values['drives']):
        p.error('Both drives must cover the measurement window')
    if any(d['waveform']=='constant' for d in base.values['drives']):
        p.error('Expected periodic input and control drives')
    if base.values['temperature']!=0:
        p.error('Initial sweep is designed for temperature=0; validate noisy runs separately')
    exe=args.exe.resolve()
    if not exe.is_file(): p.error('Executable missing: make -C src')
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    gs=np.linspace(args.g_min,args.g_max,args.points).tolist()
    cases=['both','input_only']
    if args.control_only: cases.append('control_only')
    names=['input']+[f'output_{i}' for i in base.targets('output')]
    save_json(out/'protocol.json',dict(source=str(args.config.resolve()),source_hash=fingerprint(args.config),
        engine_hash=fingerprint(exe),script_hash=fingerprint(__file__),
        baseline_control_force=base.values['drives'][1]['amplitude'],
        baseline_input_force=base.values['drives'][0]['amplitude'],
        g_values=gs, window=args.window, repeats=args.runs, cases=cases,
        dt_divisor=args.dt_divisor,seed=args.seed))
    save_json(out/'source_configuration.json',{'configuration':base.values})
    jobs=[]
    for i,g in enumerate(gs):
        for rep in range(args.runs):
            for case in cases:
                # Matched seed for all control strengths/cases within a repeat.
                jobs.append((i,g,case,rep+1,args.seed+rep,
                    out/f'point_{i:03d}'/f'repeat_{rep+1:03d}'/case))
    print(f'{len(jobs)} simulations; output: {out}',flush=True)
    results=[]
    with ThreadPoolExecutor(max_workers=args.cores) as pool:
        future_map={pool.submit(run_job,base,g,case,rep,seed,args.dt_divisor,
                    args.window,exe,folder,args.keep):(i,g,case,rep) for i,g,case,rep,seed,folder in jobs}
        for idx,future in enumerate(as_completed(future_map),1):
            result=future.result();results.append(result)
            if idx%max(1,args.runs)==0 or idx==len(jobs):
                print(f'{idx}/{len(jobs)} finished',flush=True)
    save_json(out/'replicates.json',sorted(results,key=lambda r:(r['g'],r['repeat'],r['case'])))
    rows=[]
    for g in gs:
        for case in cases:
            all_for_point=[r for r in results if r['g']==g and r['case']==case]
            good=[r for r in all_for_point if r['status']=='ok']
            for name in names:
                row=dict(g=g,control_force=g*base.values['drives'][1]['amplitude'],
                         case=case,bead=name,n_valid=len(good),n_total=args.runs)
                for key in ('R','mismatch','slips'):
                    vals=np.array([r['measurements'][name][key] for r in good],dtype=float)
                    row[key+'_mean']=float(vals.mean()) if len(vals) else ''
                    row[key+'_sem']=float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else ''
                rows.append(row)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_json(out/'summary.json',rows)
    failures=[r for r in results if r['status']!='ok']
    print(f'Finished: {len(results)-len(failures)}/{len(results)} successful. '
          f'Summary: {out}/summary.csv',flush=True)
    if failures: print(f'{len(failures)} failures; inspect replicates.json and per-run directories',flush=True)

if __name__=='__main__': main()

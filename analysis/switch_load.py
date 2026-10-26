"""Geometry-only load sweep: compare a fixed switch with an extra active rower.

All hydrodynamic interactions remain present, including feedback onto the
original switch. Success is evidence of load tolerance, not proof of a directed
5 -> 6 signal pathway. Reuse the cycle scheduler, phase extraction and classifier.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import Configuration, PER_BEAD_DEFAULTS, load_configuration, save_json, fingerprint
from run_signal_switch import sequence_repeat
from signal_switch import classify_locking
from switch_cycles import (build_schedule, refine_timestep, classify_rows,
                           aggregate, plot_trace, validate_protocol)

ROOT = Path(__file__).resolve().parents[1]


def check_source(base):
    """Require the intended baseline and resolve targets BEFORE adding bead 6."""
    if (base.n != 6 or base.targets('input') != [0]
            or base.targets('control') != [2] or base.targets('output') != [5]):
        raise ValueError('Expected six-bead baseline: input 0, control 2, output 5')
    drives = base.values['drives']
    if (not drives or len(drives) != 2
            or base.targets(drives[0]['targets']) != [0]
            or base.targets(drives[1]['targets']) != [2]):
        raise ValueError('Expected two drives targeting only beads 0 and 2')
    if any(d['waveform']=='constant' for d in drives):
        raise ValueError('Both source drives must be periodic')


def add_load(base, distance):
    """Append an unforced active rower along bead 5's positive stroke direction.

    Copy all per-bead trap parameters and angle from bead 5. Preserve every
    original reference position. Distance is reference-centre separation,
    not the instantaneous surface gap. The standard validator checks geometry.
    """
    check_source(base)
    return append_active_rower(base, distance)


def append_active_rower(base, distance):
    """Append after the last bead; monitor original output 5 and all new beads."""
    if base.n < 6:
        raise ValueError('Need the six-bead switch before appending rowers')
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('Load distance must be finite and positive')
    values = deepcopy(base.values)
    positions = base.reference_positions
    angle = np.deg2rad(values['angles_deg'][base.n-1])
    endpoint = positions[-1] + distance*np.array([np.cos(angle), np.sin(angle)])
    values['num_beads'] = base.n+1
    values['reference_x'] = [*positions[:,0].tolist(), float(endpoint[0])]
    values['reference_y'] = [*positions[:,1].tolist(), float(endpoint[1])]
    # Legacy spacing fields still need N-1 entries; explicit coordinates govern
    # the actual layout. Preserve the existing entries and append the new gap.
    values['distances'] = [*values['distances'], float(distance)]
    for key in (*PER_BEAD_DEFAULTS, 'angles_deg'):
        values[key] = [*values[key], values[key][-1]]
    # Resolve drive groups using the original config. Enlarging the output
    # monitoring group must never accidentally apply forcing to the load.
    for drive, original in zip(values['drives'], base.values['drives']):
        drive['targets'] = base.targets(original['targets'])
    values['groups']['output'] = list(range(5, base.n+1))
    values['groups']['load'] = list(range(6, base.n+1))
    result = Configuration(values)
    result.validate()
    return result


def per_bead_rows(rows, windows, criteria, bead):
    """Classify each output independently; never hide a failure in mean R."""
    selected = deepcopy(rows)
    for row in selected:
        metrics = [m for m in row['signal_metrics'] if m['bead']==bead]
        if len(metrics) != 1:
            raise ValueError(f'Missing output measurement for bead {bead}')
        row['signal_metrics'] = metrics
        row['R'] = metrics[0]['phase_consistency_R']
    return classify_rows(selected, windows, criteria)


def summarise_repeats(repeats, windows, criteria, outputs):
    result = {}
    for bead in outputs:
        selected = [per_bead_rows(rows, windows, criteria, bead) for rows in repeats]
        result[str(bead)] = aggregate(selected)
    joint = [classify_rows(deepcopy(rows), windows, criteria) for rows in repeats]
    return {'beads': result, 'joint': aggregate(joint)}


def save_window_csv(path, repeats, windows, criteria, outputs):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['repeat','bead','stage','start','stop','control_on',
                         'input_state','output_state','passed','R',
                         'relative_frequency_mismatch','slips'])
        for index, rows in enumerate(repeats, 1):
            for bead in outputs:
                for row in per_bead_rows(rows, windows, criteria, bead):
                    m = row['signal_metrics'][0]
                    writer.writerow([index,bead,row['stage'],*row['window'],
                                     row['control_on'],row['input_state'],
                                     row['output_state'],row['passed'],row['R'],
                                     m['relative_frequency_mismatch'],
                                     m['slip_candidates_total']])


def plot_layout(config, path):
    pos = config.reference_positions
    angle = np.deg2rad(config.values['angles_deg'])
    fig, ax = plt.subplots(figsize=(7,5), layout='constrained')
    ax.scatter(*pos.T)
    ax.quiver(pos[:,0],pos[:,1],np.cos(angle),np.sin(angle),
              angles='xy',scale_units='xy',scale=.5)
    for bead, (x,y) in enumerate(pos):
        ax.annotate(str(bead),(x,y),xytext=(4,5),textcoords='offset points')
    ax.set(xlabel='Reference x',ylabel='Reference y',title='Beads and stroke directions')
    ax.set_aspect('equal',adjustable='datalim')
    fig.savefig(path,dpi=150)
    plt.close(fig)


def plot_comparison(summary, path):
    factors = sorted({r['dt_divisor'] for r in summary})
    fig, axes = plt.subplots(2,len(factors),figsize=(7*len(factors),7),
                             squeeze=False,layout='constrained')
    for col,factor in enumerate(factors):
        baseline = next(r for r in summary if r['baseline'] and r['dt_divisor']==factor)
        loaded = sorted((r for r in summary if not r['baseline'] and r['dt_divisor']==factor),
                        key=lambda r:r['distance'])
        for row,state in enumerate(('on','off')):
            ax=axes[row,col]
            for bead,label in [('5','Original output (5)'),('6','Added bead (6)')]:
                ax.plot([r['distance'] for r in loaded],
                        [r['beads'][bead][state]['fraction'] for r in loaded],'o-',label=label)
            ax.axhline(baseline['beads']['5'][state]['fraction'],color='black',
                       linestyle='--',label='Six-bead baseline')
            ax.set(xlabel='Reference-centre distance from bead 5',ylabel='Passing window fraction',
                   ylim=(-.05,1.05),title=f'Control {state.upper()}, saved dt/{factor}')
            ax.legend()
    fig.savefig(path,dpi=150)
    plt.close(fig)


def run_sequence_batch(refined, spec, windows, protocol, folder, exe, runs, cores):
    """Shared continuous-run/report path for load screening and larger arrays."""
    outputs=refined.targets('output')
    repeats=[]
    with ThreadPoolExecutor(max_workers=cores) as pool:
        futures=[pool.submit(sequence_repeat,refined,spec,folder/f'repeat_{i+1}',
                             exe,protocol['seed']+i,windows) for i in range(runs)]
        for index,future in enumerate(futures,1):
            rows,t,delta=future.result()
            classify_rows(rows,windows,protocol['classification'])
            # Persist joint and individual labels alongside original metrics.
            for row in rows:
                row['output_states'] = {
                    str(m['bead']): classify_locking(m, protocol['classification'])
                    for m in row['signal_metrics']}
            save_json(folder/f'repeat_{index}'/'stages.json',rows)
            plot_trace(t,delta,windows,folder/f'phase_repeat_{index}.png',
                       labels=['Input 0',*[f'Bead {b}' for b in outputs]])
            repeats.append(rows)
            save_json(folder/'stage_metrics.json',repeats)
            print(f'{folder.name}: repeat {index}/{runs}',flush=True)
    save_window_csv(folder/'windows.csv',repeats,windows,protocol['classification'],outputs)
    return summarise_repeats(repeats,windows,protocol['classification'],outputs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,
                   default=ROOT/'results/signal_switch_pipeline/search/amplitude_3/best.json')
    p.add_argument('--experiment',type=Path,
                   default=ROOT/'results/signal_switch_pipeline/experiment.json')
    p.add_argument('--protocol',type=Path,default=ROOT/'templates/computing/switch_load.json')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--exe',type=Path,default=ROOT/'src/rowers')
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--cores',type=int,default=3)
    # These optional overrides support independent confirmation of a chosen gap
    # without hand-editing the screening template or changing the physics.
    p.add_argument('--distances',type=float,nargs='+')
    p.add_argument('--cycles',type=int)
    p.add_argument('--dt-divisors',type=int,nargs='+')
    p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.runs<1 or args.cores<1: p.error('runs and cores must be positive')
    protocol=json.loads(args.protocol.read_text())
    for key in ('distances','cycles','dt_divisors','seed'):
        value=getattr(args,key)
        if value is not None: protocol[key]=value
    # The common scheduler uses amplitude only as a saved-search selector in its
    # CLI. Here --config already selects the source, so no drive is rewritten.
    cycle_protocol={k:v for k,v in protocol.items() if k!='distances'}
    cycle_protocol['amplitude']=3
    validate_protocol(cycle_protocol)
    distances=protocol['distances']
    if not distances or len(set(distances))!=len(distances):
        p.error('Provide distinct positive distances')
    if protocol['seed']+args.runs-1>4294967295: p.error('Seed range exceeds uint32')
    base=load_configuration(args.config)
    check_source(base)
    spec=json.loads(args.experiment.read_text())
    if spec.get('objective')!='signal_switch': p.error('Requires signal_switch experiment')
    exe=args.exe.resolve()
    if not exe.is_file(): p.error('Build src/rowers first')
    # Validate every geometry and schedule before creating output or running.
    cases=[('baseline',None,base)]
    cases += [(f'load_{i+1}',float(d),add_load(base,d)) for i,d in enumerate(distances)]
    scheduled=[(label,d,*build_schedule(c,cycle_protocol)) for label,d,c in cases]
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'protocol.json',protocol)
    save_json(out/'source_configuration.json',{'configuration':base.values})
    save_json(out/'experiment.json',spec)
    save_json(out/'settings.json',{
        'runs':args.runs,'cores':args.cores,'source':str(args.config.resolve()),
        'source_sha256':fingerprint(args.config),'executable_sha256':fingerprint(exe),
        'code_sha256':{name:fingerprint(Path(__file__).with_name(name)) for name in
                      ('switch_load.py','switch_cycles.py','run_signal_switch.py',
                       'signal_switch.py','simulation.py','experiments.py','bead_analysis.py')},
        'note':'Seeds are reused across layouts, but adding a bead can change RNG '
               'consumption: this is not a guarantee of identical initial states. '
               'Windows in a repeat are correlated. Full hydrodynamic coupling retained.'})
    summary=[]
    for label,distance,config,windows in scheduled:
        outputs=config.targets('output')
        for factor in protocol['dt_divisors']:
            folder=out/label/f'dt_{factor}'
            folder.mkdir(parents=True)
            refined=refine_timestep(config,factor)
            save_json(folder/'configuration.json',{'configuration':refined.values})
            save_json(folder/'windows.json',windows)
            plot_layout(refined,folder/'layout.png')
            result = run_sequence_batch(refined,spec,windows,protocol,folder,exe,args.runs,args.cores)
            summary.append({'case':label,'baseline':distance is None,'distance':distance,
                            'dt_divisor':factor,'dt':refined.values['dt'],
                            **result})
            save_json(out/'summary.json',summary)
    plot_comparison(summary,out/'load_comparison.png')
    print(f'Finished: {out}. Screen jointly for original-output and load performance.')


if __name__=='__main__':
    main()

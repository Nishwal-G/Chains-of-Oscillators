"""Complete fixed-geometry characterisation -> device search -> validation.

All stages refuse existing output folders. Run from repository root; see
TRANSISTOR_DEVICE.md for the ordered commands and interpretation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation import load_configuration, save_json, ParameterSpace, fingerprint
from switch_cycles import refine_timestep
from transistor_device import (validate_device,check_source,actual_control,device_repeat,
    transfer,intervention,cycles,perturb_stiffness)
from transistor_measurements import valid,locked,probability,gating_contrast
from transistor_reports import (transfer_report,phase_report,cycle_report,trace_plots,write_csv)

ROOT=Path(__file__).resolve().parents[1]
TEMPLATE=ROOT/'templates/computing/transistor_device.json'
DEFAULT_SEEDS={'prepare':145250000,'optimise':145250000,'optimise-fast':145250000,'baseline':245250000,
    'validate':345250000,'gating':345350000,'amplitude':345350000,'phase':345450000,
    'cycles':345550000,'locking-map':545250000,'robustness':445250000}


def settings(args):return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}


def prepare(args):
    if args.config is None:raise ValueError('prepare requires --config pointing to your existing best.json')
    base=load_configuration(args.config)
    p=json.loads((args.protocol or TEMPLATE).read_text())
    if args.control_ratio is not None:p['control_ratio']=args.control_ratio
    check_source(base,p)
    if not 0<=args.anchor<base.n:raise ValueError('Anchor outside bead range')
    # Timings are set by each experiment; save complete original model plus
    # a fixed training seed. Control amplitude is determined by the protocol.
    start=base.updated(seed=args.seed)
    if not start.values['reference_x']:raise ValueError('Device search needs explicit planar geometry')
    spec={f'reference_{axis}[{i}]':{'bounds':[float(v-args.half_width),float(v+args.half_width)],'step':args.step}
        for axis in ('x','y') for i,v in enumerate(start.values[f'reference_{axis}']) if i!=args.anchor}
    ParameterSpace(start,spec)
    args.work.mkdir(parents=True,exist_ok=False)
    save_json(args.work/'source.json',{'configuration':base.values})
    save_json(args.work/'start.json',{'configuration':start.values})
    save_json(args.work/'device.json',p);save_json(args.work/'search.json',spec)
    save_json(args.work/'preparation.json',{'source':str(args.config.resolve()),'source_sha256':fingerprint(args.config),
        'settings':settings(args),'control_high':actual_control(base,p),
        'input_amplitude':base.values['drives'][0]['amplitude']})
    print(f'Prepared {base.n}-bead search with {len(spec)} coordinates; input force '
          f'{base.values["drives"][0]["amplitude"]:g}, high control force {actual_control(base,p):g}.')
    print('Protocol is frozen in WORK/device.json. Review before baseline/search.')


def optimise(args):
    # Keep the original optimiser, proposal rule and checkpointing.
    config=args.config or args.work/'start.json';protocol=args.protocol or args.work/'device.json'
    p=json.loads(protocol.read_text());check_source(load_configuration(config),p)
    fast_mode=args.stage=='optimise-fast'
    out=args.out or args.work/('search_fast' if fast_mode else 'search')
    command=[sys.executable,str(ROOT/'analysis/optimisation.py'),'--method','metropolis',
        '--config',str(config),'--search',str(args.work/'search.json'),'--fast-device' if fast_mode else '--device',str(protocol),
        '--exe',str(args.exe),'--iterations',str(args.iterations),'--runs',str(args.runs),
        '--cores',str(args.cores),'--seed',str(args.proposal_seed),'--set','seed',str(args.seed),
        '--search-temperature',str(args.search_temperature),'--out',str(out)]
    if args.keep:command.append('--keep')
    if fast_mode:
        from transistor_fast import validate_fast
        fast_spec=args.fast_spec or ROOT/'templates/computing/transistor_fast.json'
        fast=json.loads(fast_spec.read_text());validate_fast(fast)
        command.extend(['--fast-spec',str(fast_spec)])
        per=len(fast['fractions'])+2
    else:
        per=2*len(p['fractions'])+2*len(p['phase']['shifts'])+int(p['objective_cycles'])
        per*=1+int(p['training_disorder']>0)
    print(f'Up to {per*args.runs*(args.iterations+1)} simulations across all candidates '
          f'({per} per repeat). Invalid proposals skip evaluation.',flush=True)
    subprocess.run(command,check=True)


def source_and_protocol(args):
    source=args.config or (args.work/'start.json' if args.stage=='baseline' else args.work/'search/best.json')
    base=load_configuration(source)
    record=json.loads(source.read_text()) if source.suffix=='.json' else {}
    record=record.get('best',record)
    if args.protocol:p=json.loads(args.protocol.read_text())
    else:p=record.get('protocol',{}).get('device') or json.loads((args.work/'device.json').read_text())
    check_source(base,p)
    training=record.get('protocol',{})
    if training.get('objective') in ('weak_control_phase_device_v1','weak_control_transfer_fast_v1'):
        seeds={(base.values['seed']+i)%2**32 for i in range(training['runs'])}
        if seeds.intersection(range(args.seed,args.seed+args.runs)):
            raise ValueError('Characterisation seeds overlap search seeds; choose another --seed')
    return base,p,source


def batch(base,p,folder,args,kind='full',fractions=None,disorder=0.):
    folder.mkdir(parents=True,exist_ok=False)
    save_json(folder/'protocol.json',p);save_json(folder/'source_configuration.json',{'configuration':base.values})
    def worker(i):
        config=base.updated(seed=args.seed+i)
        if disorder:config=perturb_stiffness(config,disorder,args.seed+i+100000)
        destination=folder/f'repeat_{i}'
        destination.mkdir(parents=True)
        save_json(destination/'physical_configuration.json',{'configuration':config.values})
        if kind=='full':
            result=device_repeat(config,p,args.exe,destination,args.keep,True,True,cycle_count=p['cycles']['count'])
        elif kind=='cycles':
            result={'seed':config.values['seed'],'cycles':cycles(config,p,args.exe,destination/'cycles',args.keep,trace=True)}
        else:
            rows,cache=transfer(config,p,args.exe,destination/'transfer',args.keep,fractions)
            result={'seed':config.values['seed'],'input_bead':str(config.targets('input')[0]),'transfer':rows}
            if kind=='phase':result['phase']=intervention(config,p,args.exe,destination/'phase',cache,args.keep,True)
        save_json(destination/'result.json',result)
        return result
    repeats=[]
    with ThreadPoolExecutor(max_workers=min(args.runs,args.cores)) as pool:
        futures=[pool.submit(worker,i) for i in range(args.runs)]
        for i,future in enumerate(futures):
            repeats.append(future.result())
            save_json(folder/'results.json',repeats)
            print(f'{folder}: completed repeat {i+1}/{args.runs}',flush=True)
    if kind!='cycles':transfer_report(repeats,p,folder/'reports')
    if kind in ('full','phase'):phase_report(repeats,p,folder/'reports')
    if kind in ('full','cycles'):cycle_report(repeats,p,folder/'reports')
    trace_plots(folder,p)
    return repeats


def map_report(records,base,out):
    frequencies=sorted({r['frequency_ratio'] for r in records});controls=sorted({r['control_fraction'] for r in records})
    outputs=list(records[0]['repeats'][0]['transfer'][0]['driven']['outputs'])
    data=[]
    for row in records:
        for bead in ['input',*outputs]:
            ms=[r['transfer'][0]['driven']['input'] if bead=='input' else r['transfer'][0]['driven']['outputs'][bead] for r in row['repeats']]
            prob=probability([locked(m,row['criteria']) if valid(m) else None for m in ms])
            vals=[m['phase_consistency_R'] for m in ms if valid(m)]
            drift=[m['relative_frequency_mismatch'] for m in ms if valid(m)]
            slips=[m['slip_candidates_total'] for m in ms if valid(m)]
            data.append({'bead':bead,'frequency_ratio':row['frequency_ratio'],'control_fraction':row['control_fraction'],
                'frequency':row['frequency_ratio']/base.values['drives'][0]['period'],
                'R':float(np.mean(vals)) if vals else None,'mismatch':float(np.mean(drift)) if drift else None,
                'slips':float(np.mean(slips)) if slips else None,**prob})
    for bead in ['input',*outputs]:
        fig,axes=plt.subplots(1,4,figsize=(17,4),layout='constrained')
        for ax,key in zip(axes,('fraction_all','R','mismatch','slips')):
            grid=np.full((len(controls),len(frequencies)),np.nan)
            for r in data:
                if r['bead']==bead:grid[controls.index(r['control_fraction']),frequencies.index(r['frequency_ratio'])]=np.nan if r[key] is None else r[key]
            # Categorical centres avoid misleading area/extent at irregular grids.
            im=ax.imshow(grid,origin='lower',aspect='auto',interpolation='nearest',
                **({'vmin':0,'vmax':1} if key in ('fraction_all','R') else {}))
            ax.set(xticks=range(len(frequencies)),xticklabels=[f'{v:g}' for v in frequencies],
                   yticks=range(len(controls)),yticklabels=[f'{v:g}' for v in controls],
                   xlabel='Input frequency / saved input frequency',ylabel='Control / high-control force',title=key)
            fig.colorbar(im,ax=ax)
        fig.suptitle(f'Bead {bead}; control frequency is fixed')
        fig.savefig(out/f'locking_map_{bead}.png',dpi=150);plt.close(fig)
    save_json(out/'map_summary.json',data)


def run_analysis(args):
    base,p,source=source_and_protocol(args)
    out=args.out or args.work/args.stage
    if not args.exe.is_file():raise ValueError('Build src/rowers first')
    factors=args.dt_divisors or ([1,2] if args.stage=='validate' else [1])
    dense=sorted(p['fractions']+[float(v) for v in np.linspace(0,1,args.points)
        if not np.any(np.isclose(v,p['fractions'],rtol=0,atol=1e-12))])
    # Reject undersampled map clocks before any expensive run begins.
    if args.stage=='locking-map':
        for ratio in p['map']['frequency_ratios']:
            candidate=base.updated(**{'drives[0].period':base.values['drives'][0]['period']/ratio})
            check_source(candidate,p)
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'settings.json',{'command':settings(args),'source':str(source.resolve()),
        'source_sha256':fingerprint(source),'protocol':p,'executable_sha256':fingerprint(args.exe),
        'code_sha256':{n:fingerprint(ROOT/'analysis'/n) for n in ('run_transistor_suite.py','transistor_device.py',
            'transistor_measurements.py','transistor_reports.py','simulation.py','bead_analysis.py','signal_switch.py','diagnose_switch.py')}})
    comparison=[]
    for factor in factors:
        config=refine_timestep(base,factor);folder=out/f'dt_{factor}';folder.mkdir()
        if args.stage in ('baseline','validate'):
            repeats=batch(config,p,folder/'full',args)
            comparison.append({'dt_divisor':factor,'dt':config.values['dt'],
                'score_mean':float(np.mean([r['score'] for r in repeats])),
                'score_sem':float(np.std([r['score'] for r in repeats],ddof=1)/np.sqrt(args.runs)) if args.runs>1 else None,
                'note':'Nominal characterisation, configured cycle count; not the nominal-plus-disorder one-cycle search score.'})
            save_json(out/'comparison.json',comparison)
            if args.stage=='validate':batch(config,p,folder/'dense',args,'gating',dense)
        elif args.stage in ('gating','amplitude','phase','cycles'):
            batch(config,p,folder/'measurements',args,args.stage,dense if args.stage in ('gating','amplitude') else [0.,1.] if args.stage=='phase' else None)
        elif args.stage=='locking-map':
            records=[]
            for i,ratio in enumerate(p['map']['frequency_ratios']):
                current=config.updated(**{'drives[0].period':config.values['drives'][0]['period']/ratio})
                for j,fraction in enumerate(p['map']['control_fractions']):
                    repeats=batch(current,p,folder/f'frequency_{i}_control_{j}',args,'gating',[fraction])
                    records.append({'frequency_ratio':ratio,'control_fraction':fraction,'criteria':p['criteria'],'repeats':repeats})
                    save_json(folder/'map_results.json',records)
            map_report(records,config,folder)
        else:
            summary=[]
            for i,disorder in enumerate(p['robustness']['stiffness_fractions']):
                for j,offset in enumerate(p['robustness']['temperature_offsets']):
                    temperature=base.values['temperature']+offset
                    current=config.updated(temperature=temperature)
                    repeats=batch(current,p,folder/f'stiffness_{i}_temperature_{j}',args,'gating',[0.,1.],disorder)
                    for bead in map(str,config.targets('output')):
                        flags=[]
                        for r in repeats:
                            low,high=r['transfer'];ms=[v[case]['outputs'][bead] for v in (low,high) for case in ('driven','control_only')]
                            if not all(valid(m) for m in ms) or not all(valid(v['driven']['input']) for v in (low,high)):flags.append(None);continue
                            flags.append(locked(high['driven']['outputs'][bead],p['criteria']) and
                                all(locked(v['driven']['input'],p['criteria']) for v in (low,high)) and
                                low['driven']['outputs'][bead]['phase_consistency_R']<=p['criteria']['off_max'] and
                                all(v['control_only']['outputs'][bead]['phase_consistency_R']<=p['criteria']['off_max'] for v in (low,high)))
                        summary.append({'bead':bead,'stiffness_fraction':disorder,'temperature':temperature,**probability(flags)})
                    save_json(folder/'robustness_summary.json',summary)
            for bead in map(str,config.targets('output')):
                fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
                for disorder in p['robustness']['stiffness_fractions']:
                    selected=[r for r in summary if r['bead']==bead and r['stiffness_fraction']==disorder]
                    ax.plot([r['temperature'] for r in selected],[r['fraction_all'] for r in selected],'o-',label=f'Stiffness +/- {100*disorder:g}%')
                ax.set(xlabel='Simulator temperature',ylabel='All gating targets / all repeats',ylim=(-.03,1.03));ax.legend()
                fig.savefig(folder/f'robustness_{bead}.png',dpi=150);plt.close(fig)
    print(f'Finished {args.stage}: {out}. Inspect invalid counts and each output; no automatic transistor claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=list(DEFAULT_SEEDS))
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--fast-spec',type=Path,help='optimise-fast grid/weights; default templates/computing/transistor_fast.json')
    p.add_argument('--out',type=Path)
    p.add_argument('--exe',type=Path,default=Path('src/rowers'))
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--cores',type=int,default=3)
    p.add_argument('--seed',type=int)
    p.add_argument('--iterations',type=int,default=300)
    p.add_argument('--proposal-seed',type=int,default=42)
    p.add_argument('--search-temperature',type=float,default=.05)
    p.add_argument('--points',type=int,default=21)
    p.add_argument('--dt-divisors',type=int,nargs='+')
    p.add_argument('--control-ratio',type=float,help='prepare only; default comes from protocol')
    p.add_argument('--anchor',type=int,default=4)
    p.add_argument('--half-width',type=float,default=6.)
    p.add_argument('--step',type=float,default=.5)
    p.add_argument('--keep',action='store_true',help='Keep bulky raw trajectories')
    args=p.parse_args();args.exe=args.exe.resolve()
    if args.seed is None:args.seed=DEFAULT_SEEDS[args.stage]
    if (args.runs<1 or args.cores<1 or args.iterations<0 or args.points<3 or args.seed<0 or args.seed+args.runs>2**32 or
        not np.all(np.isfinite([args.search_temperature,args.half_width,args.step])) or
        min(args.search_temperature,args.half_width,args.step)<=0):p.error('Invalid numerical settings')
    if args.dt_divisors and (any(f<1 for f in args.dt_divisors) or len(set(args.dt_divisors))!=len(args.dt_divisors)):p.error('Invalid dt divisors')
    if args.fast_spec is not None and args.stage!='optimise-fast':p.error('--fast-spec applies only to optimise-fast')
    if args.control_ratio is not None and args.stage!='prepare':p.error('--control-ratio is preparation-only; later stages use the saved protocol')
    if args.stage=='prepare':prepare(args)
    elif args.stage in ('optimise','optimise-fast'):optimise(args)
    else:run_analysis(args)

if __name__=='__main__':main()

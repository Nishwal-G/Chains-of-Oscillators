"""Confirm drives, search geometry, validate, and test single/repeated switching.

Use the same work folder for successive stages. Each stage refuses to overwrite
previous results. The runner calls the existing optimiser; only signal scoring
is new. A saved search best is exploratory until independently validated.
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
from simulation import load_configuration, save_json, run_simulation, ParameterSpace, fingerprint
from diagnose_switch import prepare_config, validate_protocol, phase_analyser, ROOT
from experiments import ExperimentEvaluator, plot_results, measure_window, validate_experiment
from sweep_switch_drive import driven_config

TEMPLATES = ROOT/'templates/computing'


def evaluate(config, spec, out, exe, runs, cores):
    """All execution, seeds, scoring and metrics remain in the shared evaluator."""
    out.mkdir(parents=True,exist_ok=False)
    evaluator=ExperimentEvaluator(exe,out,runs,cores,experiment=spec)
    result=evaluator.evaluate(config)
    save_json(out/'result.json',result)
    plot_results(result,config,out)
    return {'directory':str(out),'score':result['score'],'score_sem':result['score_sem'],
            'input_success_fraction':result['input_success_fraction'],
            'switch_successes':result['switch_successes'],'switch_trials':result['switch_trials']}


def timed(config, total, seed):
    drives=deepcopy(config.values['drives'])
    for d in drives: d['stop']=total
    return config.updated(total_time=total,seed=seed,drives=drives)


def sequence_repeat(config, spec, out, exe, seed, schedule=None):
    """One continuous trajectory: retain state at control transitions.

    Offline crossing phases are used; stage windows avoid transitions. No
    causal latency is inferred from interpolated phases across a transition.
    """
    out.mkdir(parents=True)
    # Optional schedule reuses the same simulation and phase-analysis path.
    if schedule is None:
        config=timed(config,200,seed)
        input_drive,control=deepcopy(config.values['drives'])
        first=deepcopy(control); first['stop']=65
        last=deepcopy(control); last.update(start=130,stop=200)
        config=config.updated(drives=[input_drive,first,last])
        windows=[('on_1',45,60),('off',105,120),('on_2',175,190)]
    else:
        config=config.updated(seed=seed)
        input_drive=config.values['drives'][0]
        windows=[(r['stage'], *r['window']) for r in schedule]
    save_json(out/'configuration.json',{'configuration':config.values})
    reader=run_simulation(config,exe,out/'simulation')
    analyser=phase_analyser(config,reader)
    rows=[]
    for name,start,stop in windows:
        row,_=measure_window(analyser,config,spec,config.targets('output'),config.targets('input')[0],start,stop,np.pi/4)
        rows.append({'stage':name,**row})
    t=np.asarray(analyser.time_series)
    clock=2*np.pi*t/input_drive['period']+input_drive['phase']
    delta=np.column_stack([(analyser._get_phase(config.targets(g)[0])-clock)/(2*np.pi)
                           for g in ('input','output')])
    np.savez_compressed(out/'phase_trace.npz',time=t,input_minus_drive=delta[:,0],output_minus_drive=delta[:,1])
    save_json(out/'stages.json',rows)
    # Raw data are no longer needed; exact configuration and phase traces remain.
    import shutil
    shutil.rmtree(out/'simulation')
    return rows,t,delta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['confirm','search','validate','sequence','cycles'],required=True)
    p.add_argument('--work',type=Path,default=ROOT/'results/signal_switch_pipeline')
    p.add_argument('--config',type=Path,default=TEMPLATES/'junction_diagnostic.json')
    p.add_argument('--exe',type=Path,default=ROOT/'src/rowers')
    p.add_argument('--cores',type=int,default=3)
    p.add_argument('--runs',type=int,default=10,help='Repeats per confirmation/validation/sequence test')
    p.add_argument('--search-runs',type=int,default=3)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--perturbations',type=int,default=2)
    p.add_argument('--position-jitter',type=float,default=.2,help='Uniform +/- displacement in coordinate units')
    p.add_argument('--cycle-protocol',type=Path,default=TEMPLATES/'switch_cycles.json')
    p.add_argument('--cycle-out',type=Path,default=None,
                   help='Fresh cycle output folder; defaults to WORK/cycles')
    args=p.parse_args()
    if min(args.cores,args.runs,args.search_runs)<1 or args.iterations<0 or args.perturbations<0:
        p.error('Invalid repeat/iteration counts')
    if not np.isfinite(args.position_jitter) or args.position_jitter<0:
        p.error('position-jitter must be finite and nonnegative')
    exe=args.exe.resolve()
    if not exe.is_file(): p.error('Missing src/rowers; run make -C src')
    if args.stage=='cycles':
        from switch_cycles import run_cycles
        run_cycles(args, sequence_repeat)
        return
    work=args.work.resolve(); work.mkdir(parents=True,exist_ok=True)
    # Freeze configuration, protocol and free coordinates on the first stage.
    # Later stages use this snapshot, even if templates are edited meanwhile.
    if args.stage=='confirm':
        if (work/'base.json').exists(): p.error('Work folder already initialised; choose a fresh --work')
        base=load_configuration(args.config)
        validate_protocol(base,[60,95])
        if base.n != 6 or base.targets('input') != [0] or base.targets('control') != [2] or base.targets('output') != [5]:
            p.error('Runner expects the six-bead input=0, control=2, output=5 layout')
        spec=json.loads((TEMPLATES/'control_signal_switch.json').read_text())
        search=json.loads((TEMPLATES/'search_geometry_offset.json').read_text())
        ParameterSpace(base,search)
        save_json(work/'base.json',{'configuration':base.values})
        save_json(work/'experiment.json',spec); save_json(work/'search.json',search)
    else:
        if not (work/'base.json').is_file(): p.error('Run --stage confirm first in this work folder')
        base=load_configuration(work/'base.json')
        spec=json.loads((work/'experiment.json').read_text())
        search=json.loads((work/'search.json').read_text())
    validate_experiment(spec)
    stage=work/args.stage; stage.mkdir(exist_ok=False)
    save_json(stage/'settings.json',{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'executable_sha256':fingerprint(exe),'runner_sha256':fingerprint(__file__)})
    summary=[]
    for amplitude in (2,3):
        label=f'amplitude_{amplitude}'
        # Use amplitudes selected from the sweep; retain the exact saved geometry.
        source=driven_config(base,amplitude,.184715,'input-only')
        if args.stage=='confirm':
            for factor in (2,4):
                config=prepare_config(timed(source,100,750000),'both',factor,750000)
                row=evaluate(config,spec,stage/f'{label}_dt{factor}',exe,args.runs,args.cores)
                summary.append({'amplitude':amplitude,'dt_divisor':factor,**row})
        elif args.stage=='search':
            # Physical intervals stay fixed; refinement is relative to ORIGINAL
            # base.json, never relative to an already-refined best.json.
            config=prepare_config(timed(source,100,850000),'both',4,850000)
            path=stage/f'{label}.json'; save_json(path,{'configuration':config.values})
            dest=stage/label
            subprocess.run([sys.executable,str(ROOT/'analysis/optimisation.py'),
                '--config',str(path),'--experiment',str(work/'experiment.json'),
                '--search',str(work/'search.json'),'--exe',str(exe),
                '--method','metropolis','--iterations',str(args.iterations),
                '--runs',str(args.search_runs),'--cores',str(args.cores),
                '--search-temperature','.05','--seed',str(42+amplitude),'--out',str(dest)],check=True)
            result=json.loads((dest/'best.json').read_text())
            summary.append({'amplitude':amplitude,'score':result['score'],'best_file':str(dest/'best.json')})
        else:
            path=work/'search'/label/'best.json'
            if not path.is_file(): p.error(f'Missing {path}; complete --stage search first')
            config=timed(load_configuration(path),200,950000)
            if args.stage=='validate':
                validation=deepcopy(spec)
                for case in validation['cases']:
                    case['window']=[120,190]
                    case['windows']=[[60,95]]
                # Validate selected geometry at its saved dt (original dt/4),
                # then at dt/8. No extra division is applied to the first run.
                for factor in (1,2):
                    refined=prepare_config(config,'both',factor,950000)
                    summary.append({'amplitude':amplitude,'variant':f'saved_dt_div_{factor}',
                        **evaluate(refined,validation,stage/f'{label}_dt{factor}',exe,args.runs,args.cores)})
                rng=np.random.default_rng(1050000+amplitude)
                for number in range(args.perturbations):
                    for attempt in range(100):
                        values=deepcopy(config.values)
                        # Perturb just the seven searched coordinates; the anchor,
                        # stroke angles and all trap parameters remain fixed.
                        for key in search:
                            name,index=key[:-1].split('[')
                            values[name][int(index)]+=rng.uniform(-args.position_jitter,args.position_jitter)
                        try:
                            perturbed=config.updated(**values)
                            break
                        except ValueError:
                            continue  # Reject overlapping/invalid geometries.
                    else: raise ValueError('Unable to generate a valid position perturbation')
                    summary.append({'amplitude':amplitude,'variant':f'perturbation_{number+1}',
                        **evaluate(perturbed,validation,stage/f'{label}_jitter_{number+1}',exe,args.runs,args.cores)})
            else:
                with ThreadPoolExecutor(max_workers=args.cores) as pool:
                    futures=[pool.submit(sequence_repeat,config,spec,stage/label/f'repeat_{i+1}',exe,1150000+i)
                             for i in range(args.runs)]
                    repeats=[f.result() for f in futures]
                for i,(rows,t,delta) in enumerate(repeats,1):
                    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
                    for col,ax in enumerate(axes):
                        valid=np.isfinite(delta[:,col])
                        ax.plot(t[valid],delta[valid,col]-delta[valid,col][0])
                        ax.axvspan(65,130,color='grey',alpha=.2,label='Control off')
                        ax.set_ylabel(('Input' if col==0 else 'Output')+' − drive (cycles)')
                        ax.legend()
                    axes[-1].set_xlabel('Simulation time')
                    fig.savefig(stage/label/f'phase_repeat_{i}.png',dpi=150); plt.close(fig)
                summary.append({'amplitude':amplitude,'stage_metrics':[r[0] for r in repeats],
                                'note':'Continuous trajectory; offline phases do not define causal latency'})
        save_json(stage/'summary.json',summary)
        print(f'Completed {args.stage}, amplitude {amplitude}; {stage}',flush=True)
    print('Inspect summary.json and per-case reports before advancing. No automatic success/convergence claim.',flush=True)


if __name__=='__main__': main()

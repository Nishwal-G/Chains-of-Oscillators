"""Prepare, search and independently validate seven/eight-bead switches.

This is orchestration only: reuse the existing Metropolis optimiser, simulator,
four-case evaluator, cycle scheduler, per-output classifier and plotting.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import load_configuration, ParameterSpace, save_json, fingerprint
from experiments import ExperimentEvaluator, validate_experiment, plot_results
from run_signal_switch import timed
from diagnose_switch import prepare_config
from switch_load import check_source, append_active_rower, run_sequence_batch, plot_layout
from switch_cycles import build_schedule, refine_timestep

ROOT=Path(__file__).resolve().parents[1]
TEMPLATES=ROOT/'templates/computing'


def prepare_arrays(base, count, geometry):
    """Generate exact geometry-only search JSON and three load-spacing starts."""
    check_source(base)
    if count not in (7,8): raise ValueError('Choose seven or eight beads')
    if abs(base.values['angles_deg'][5] % 360)>1e-10:
        raise ValueError('This geometry template expects a horizontal output stroke')
    search={}
    for name, setting in geometry['original_coordinates'].items():
        if not name.startswith(('reference_x[','reference_y[')) or name in ('reference_x[4]','reference_y[4]'):
            raise ValueError('Only reference coordinates are free; bead 4 is the fixed anchor')
        centre=base.get(name)
        half=setting['half_width']
        if not np.isfinite(half) or half<=0: raise ValueError('Positive coordinate half_width required')
        search[name]={'bounds':[centre-half,centre+half],'step':setting['step']}
    gaps=geometry['initial_load_gaps']
    if not gaps or any(not np.isfinite(g) or g<=0 for g in gaps):
        raise ValueError('Positive starting load gaps required')
    x5,y5=base.reference_positions[5]
    for bead in range(6,count):
        search[f'reference_x[{bead}]']={
            'bounds':[x5+geometry['load_x_min_offset'],
                      x5+(count-6)*max(gaps)+geometry['load_x_padding']],
            'step':geometry['load_step']}
        search[f'reference_y[{bead}]']={
            'bounds':[y5-geometry['load_y_half_width'],y5+geometry['load_y_half_width']],
            'step':geometry['load_step']}
    starts=[]
    for gap in gaps:
        config=base
        while config.n<count: config=append_active_rower(config,gap)
        # Preserve the saved timestep; cap output sampling using the shared rule.
        # Same simulator seeds across searches make scores comparable; proposal
        # RNG seeds differ. Fresh seeds are reserved for validation.
        config=prepare_config(timed(config,100,5250000),'both',1,5250000)
        ParameterSpace(config,search)
        starts.append(config)
    return starts,search


def prepare(args):
    base=load_configuration(args.config)
    geometry=json.loads(args.geometry.read_text())
    starts,search=prepare_arrays(base,args.beads,geometry)
    spec=json.loads((TEMPLATES/'control_slip_switch.json').read_text())
    spec['period']=base.values['drives'][0]['period']
    validate_experiment(spec)
    args.work.mkdir(parents=True,exist_ok=False)
    save_json(args.work/'source_configuration.json',{'configuration':base.values})
    save_json(args.work/'geometry_template.json',geometry)
    save_json(args.work/'experiment.json',spec)
    save_json(args.work/'search.json',search)
    save_json(args.work/'preparation.json',{
        'beads':args.beads,'starts':len(starts),'source':str(args.config.resolve()),
        'source_sha256':fingerprint(args.config),'generator_sha256':fingerprint(__file__)})
    for index,config in enumerate(starts,1):
        save_json(args.work/f'start_{index}.json',{'configuration':config.values})
    print(f'Prepared {len(starts)} starts for {args.beads} beads in {args.work}')


def search(args):
    preparation=json.loads((args.work/'preparation.json').read_text())
    starts=args.starts or list(range(1,preparation['starts']+1))
    if any(i<1 or i>preparation['starts'] for i in starts) or len(set(starts))!=len(starts):
        raise ValueError('Invalid start indices')
    root=args.work/'search'
    root.mkdir(exist_ok=True)
    # Do not partially start a requested batch if one destination already exists.
    if any((root/f'start_{i}').exists() for i in starts):
        raise ValueError('Search destination exists; choose only unstarted --starts indices')
    for index in starts:
        subprocess.run([
            sys.executable,str(ROOT/'analysis/optimisation.py'),
            '--config',str(args.work/f'start_{index}.json'),
            '--experiment',str(args.work/'experiment.json'),
            '--search',str(args.work/'search.json'),'--exe',str(args.exe),
            '--method','metropolis','--iterations',str(args.iterations),
            '--runs',str(args.runs),'--cores',str(args.cores),
            '--search-temperature',str(args.search_temperature),
            '--seed',str(4200+index),'--out',str(root/f'start_{index}')
        ],check=True)
        # Inspect all available search checkpoints, including previously finished
        # starts. Selection is exploratory; never claim validation from search.
        rows=[]
        for path in sorted(root.glob('start_*/best.json')):
            record=json.loads(path.read_text())
            if record and record.get('status')=='ok':
                rows.append({'start':path.parent.name,'score':record['score'],'path':str(path)})
        save_json(args.work/'search_summary.json',rows)
        best=max(rows,key=lambda row:row['score'])
        selected=json.loads(Path(best['path']).read_text())
        save_json(args.work/'selected_best.json',selected)
        save_json(args.work/'selection.json',best)
        print(f"Best observed score {best['score']:.6f}, from {best['start']}",flush=True)


def validate(args):
    selected=args.selected or args.work/'selected_best.json'
    base=load_configuration(selected)
    if base.n not in (7,8) or base.targets('output')!=list(range(5,base.n)):
        raise ValueError('Selected configuration must monitor bead 5 and every added bead')
    spec=json.loads((args.work/'experiment.json').read_text())
    validate_experiment(spec)
    root=args.out or args.work/'validation'
    root.mkdir(parents=True,exist_ok=False)
    save_json(root/'source_configuration.json',{'configuration':base.values})
    save_json(root/'settings.json',{
        'selected':str(selected),'selected_sha256':fingerprint(selected),
        'runs':args.runs,'cores':args.cores,'cycles':args.cycles,
        'executable_sha256':fingerprint(args.exe),'runner_sha256':fingerprint(__file__),
        'code_sha256':{name:fingerprint(Path(__file__).with_name(name)) for name in
          ('switch_load.py','switch_cycles.py','run_signal_switch.py','signal_switch.py',
           'experiments.py','simulation.py','bead_analysis.py')}})
    stationary=deepcopy(spec)
    for case in stationary['cases']: case['window']=[120,190]
    save_json(root/'stationary_experiment.json',stationary)
    criteria={k:spec[k] for k in ('on_min','frequency_tolerance',
                                'slipping_frequency_min','slipping_min_slips')}
    cycle_protocol=dict(amplitude=3,cycles=args.cycles,on_duration=65,off_duration=65,
                        settling_time=45,measurement_duration=15,dt_divisors=[1,2],
                        seed=7250000,classification=criteria)
    save_json(root/'cycle_protocol.json',cycle_protocol)
    continuous,windows=build_schedule(base,cycle_protocol)
    summary=[]
    for factor in (1,2):
        folder=root/f'dt_{factor}';folder.mkdir()
        config=refine_timestep(timed(base,200,6250000),factor)
        evaluator=ExperimentEvaluator(args.exe,folder/'stationary',args.runs,args.cores,
                                      experiment=stationary)
        record=evaluator.evaluate(config)
        save_json(folder/'stationary'/'result.json',record)
        plot_results(record,config,folder/'stationary')
        cycles=folder/'cycles';cycles.mkdir()
        refined=refine_timestep(continuous,factor)
        save_json(cycles/'configuration.json',{'configuration':refined.values})
        save_json(cycles/'windows.json',windows)
        plot_layout(refined,cycles/'layout.png')
        results=run_sequence_batch(refined,spec,windows,cycle_protocol,cycles,
                                   args.exe,args.runs,args.cores)
        summary.append({'dt_divisor':factor,'dt':refined.values['dt'],
                        'stationary_successes':record['switch_successes'],
                        'stationary_trials':record['switch_trials'],
                        'stationary_score':record['score'],'cycles':results})
        save_json(root/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,state in zip(axes,('on','off')):
        for bead in base.targets('output'):
            ax.plot([1,2],[r['cycles']['beads'][str(bead)][state]['fraction'] for r in summary],
                    'o-',label=f'Bead {bead}')
        ax.plot([1,2],[r['cycles']['joint'][state]['fraction'] for r in summary],
                'k--',label='All outputs jointly')
        ax.set(xticks=[1,2],xticklabels=['Saved dt','Saved dt/2'],
               ylim=(-.05,1.05),ylabel='Passing window fraction',title='Control '+state.upper())
        ax.legend()
    fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    print(f'Validation complete: {root}. Inspect every bead and both timesteps.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','search','validate'],required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--config',type=Path,
                        default=ROOT/'results/signal_switch_pipeline/search/amplitude_3/best.json')
    parser.add_argument('--beads',type=int,choices=[7,8],default=7)
    parser.add_argument('--geometry',type=Path,default=TEMPLATES/'larger_geometry.json')
    parser.add_argument('--starts',type=int,nargs='+')
    parser.add_argument('--exe',type=Path,default=ROOT/'src/rowers')
    parser.add_argument('--iterations',type=int,default=300)
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--cores',type=int,default=3)
    parser.add_argument('--search-temperature',type=float,default=.05)
    parser.add_argument('--cycles',type=int,default=5)
    parser.add_argument('--selected',type=Path,help='Optional particular search best.json for validation')
    parser.add_argument('--out',type=Path,help='Fresh validation folder override')
    args=parser.parse_args()
    args.work=args.work.resolve();args.exe=args.exe.resolve()
    if args.runs<1 or args.cores<1 or args.iterations<0 or args.cycles<1:
        parser.error('Invalid counts')
    if not np.isfinite(args.search_temperature) or args.search_temperature<=0:
        parser.error('Positive finite search temperature required')
    if args.stage!='prepare' and not args.exe.is_file(): parser.error('Build src/rowers first')
    {'prepare':prepare,'search':search,'validate':validate}[args.stage](args)


if __name__=='__main__':
    main()

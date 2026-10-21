"""Prepare offset layouts, reuse the optimiser, then validate the best candidates.

No new dynamics or scoring is implemented here. All configurations and paths
are saved beside the results so no hand-written follow-up commands are needed.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from simulation import Configuration, ParameterSpace, load_configuration, save_json
from experiments import validate_experiment

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / 'templates/computing'


def prepare_starts(bases, specification):
    """Keep traps and stroke angles; vary only control-branch placement initially."""
    starts = []
    for i, path in enumerate(bases, 1):
        base = load_configuration(path)
        if base.n != 6 or base.values['drives'] is None or len(base.values['drives']) != 2:
            raise ValueError('Expected a six-bead configuration with two explicit drives')
        # This protocol measures bead 5 relative to bead 0; control drives bead 2.
        if base.targets('input') != [0] or base.targets('control') != [2] or base.targets('output') != [5]:
            raise ValueError('Expected groups input=[0], control=[2], output=[5]')
        if base.targets(base.values['drives'][0]['targets']) != [0] or base.targets(base.values['drives'][1]['targets']) != [2]:
            raise ValueError('Expected drive 0 on input and drive 1 on control')
        for label, x2, x3 in [('centre',0,0), ('left',-8,-4), ('right',8,4)]:
            values = deepcopy(base.values)
            # Explicitly select the agreed forcing; do not optimise amplitudes.
            values.update(total_time=60, seed=12345)
            for drive, amplitude in zip(values['drives'], [.5,5]):
                drive.update(amplitude=amplitude, stop=60)
            values['reference_x'][2:4] = [x2,x3]
            if any(values['reference_y'][j] != 0 for j in (0,1,4,5)) or values['reference_x'][4] != 0:
                raise ValueError('Base must have horizontal input/output arms and bead 4 at the origin')
            config = Configuration(values)
            # Reuse existing bounds and pair-separation validation. Never clamp
            # an out-of-range saved candidate into a different starting layout.
            ParameterSpace(config, specification)
            starts.append((f'base_{i}_{label}', str(Path(path).resolve()), config))
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bases', nargs='+', type=Path,
                        help='Saved best.json files; each generates three offset starts')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--cores', type=int, default=3)
    parser.add_argument('--validation-runs', type=int, default=50)
    parser.add_argument('--validate-top', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if min(args.iterations,args.runs,args.cores,args.validation_runs,args.validate_top) < 1:
        parser.error('Counts must be positive')
    bases = args.bases or [TEMPLATES/'junction_switch_long.json']
    spec = json.loads((TEMPLATES/'search_geometry_offset.json').read_text())
    starts = prepare_starts(bases,spec)
    for name in ('control_switch_long.json','control_switch_validation.json'):
        validate_experiment(json.loads((TEMPLATES/name).read_text()))
    exe = ROOT/'src/rowers'
    if not args.prepare_only and not exe.is_file():
        parser.error('Missing src/rowers. Run make -C src first.')
    out = (args.out or ROOT/'results'/('geometry_offset_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'search.json',spec)
    manifest=[]
    for i,(label,source,config) in enumerate(starts):
        path=out/f'{label}.json'
        save_json(path,{'configuration':config.values})
        manifest.append({'label':label,'source':source,'config':str(path),'proposal_seed':42+137*i})
    save_json(out/'starts.json',manifest)
    print(f'Prepared {len(starts)} starts in {out}',flush=True)
    if args.prepare_only:
        return
    searches=[]
    for item in manifest:
        dest=out/item['label']
        subprocess.run([sys.executable,str(ROOT/'analysis/optimisation.py'),
            '--config',item['config'],'--experiment',str(TEMPLATES/'control_switch_long.json'),
            '--search',str(out/'search.json'),'--exe',str(exe),
            '--method','metropolis','--iterations',str(args.iterations),
            '--runs',str(args.runs),'--cores',str(args.cores),
            '--search-temperature','0.05','--seed',str(item['proposal_seed']),
            '--out',str(dest)],check=True,cwd=ROOT)
        best=dest/'best.json'
        record=json.loads(best.read_text())
        searches.append({'label':item['label'],'score':record['score'],'best_file':str(best)})
        save_json(out/'search_summary.json',searches)
    # Select by training score only; fresh seeds are reserved for validation.
    validations=[]
    for item in sorted(searches,key=lambda row:row['score'],reverse=True)[:args.validate_top]:
        config=load_configuration(item['best_file'])
        values=deepcopy(config.values)
        values.update(total_time=100,seed=550000)
        for drive in values['drives']:
            drive['stop']=100
        path=out/f"{item['label']}_validation_config.json"
        save_json(path,{'configuration':Configuration(values).values})
        dest=out/f"{item['label']}_validation"
        subprocess.run([sys.executable,str(ROOT/'analysis/experiments.py'),
            '--config',str(path),'--experiment',str(TEMPLATES/'control_switch_validation.json'),
            '--exe',str(exe),'--runs',str(args.validation_runs),'--cores',str(args.cores),
            '--out',str(dest)],check=True,cwd=ROOT)
        record=json.loads((dest/'result.json').read_text())
        row={'label':item['label'],'result_file':str(dest/'result.json')}
        for key in ('score','score_sem','switch_successes','switch_trials','off_loss','on_loss','control_penalty'):
            row[key]=record[key]
        for case in ('neither','input','control','both'):
            row[case+'_R']=sum(rep['cases'][case]['R'] for rep in record['replicates'])/len(record['replicates'])
        validations.append(row)
        save_json(out/'validation_summary.json',validations)
        print(json.dumps(row,indent=2),flush=True)
    print(f'Finished. Send: {out / "validation_summary.json"}',flush=True)


if __name__ == '__main__':
    main()

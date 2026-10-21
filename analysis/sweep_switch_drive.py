"""Calibrate forcing at a fixed geometry before resuming geometry optimisation.

All dynamics and phase measurements come from diagnose_switch.py. This grid
search adds only orchestration, explicit screening criteria and plots. It does
not change the existing optimiser objective or any trap parameter.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation import load_configuration, save_json, fingerprint
from diagnose_switch import ROOT, CASE_SCALES, prepare_config, diagnose, validate_protocol


def select_source(config=None, summary=None):
    """Use the saved candidate, or select highest score in an offset validation summary.

    Scores are from the OLD mutual-coherence objective; this selects a starting
    geometry only, without asserting that it transmits a signal. Explicitly
    supplied missing files cause an error rather than a silent template fallback.
    """
    if summary is None:
        path = config or ROOT/'templates/computing/junction_diagnostic.json'
        return path.resolve(), 'Explicit configuration' if config else 'Saved base_1_left candidate'
    rows = json.loads(summary.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty offset validation_summary.json list')
    if any(not np.isfinite(float(r['score'])) for r in rows):
        raise ValueError('Validation scores must be finite')
    best = max(rows, key=lambda r: float(r['score']))
    saved = Path(best['result_file'])
    # Support moving the whole results folder between computers. Prefer the
    # selected summary's sibling validation folder over an old absolute path.
    sibling = summary.resolve().parent / (str(best['label'])+'_validation') / 'result.json'
    choices = [sibling, saved if saved.is_absolute() else summary.resolve().parent/saved]
    path = next((x for x in choices if x.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Selected {best['label']} but result.json is missing. Pass --config with its saved configuration instead.")
    return path.resolve(), f"{best['label']}: highest previous validation score {best['score']}"


def driven_config(base, amplitude, period, period_mode):
    """Vary drive force, not bead stroke amplitude or trap stiffness."""
    drives = deepcopy(base.values['drives'])
    drives[0].update(amplitude=float(amplitude), period=float(period))
    if period_mode == 'shared':
        drives[1]['period'] = float(period)
    return base.updated(drives=drives)


def is_locked(metric, period, on_min, frequency_tolerance, max_slips):
    """Require coherence AND frequency agreement AND few slips in each repeat.

    frequency_tolerance is dimensionless: |f_bead-f_drive|/f_drive.
    These are declared screening thresholds, not universal physical constants.
    """
    return (metric['R'] >= on_min and
            abs(metric['frequency_mismatch'])*period <= frequency_tolerance and
            metric['slip_count'] <= max_slips)


def summarise(repeats, period, on_min, off_max, frequency_tolerance, max_slips):
    """Count passes per matched repeat; do not threshold ensemble mean bars."""
    valid = [r for r in repeats if r['status'] == 'ok']
    inputs, switches, qualities, contrasts = [], [], [], []
    for rep in valid:
        cases = rep['cases']
        locks = [is_locked(cases[c]['input_drive'], period, on_min,
                           frequency_tolerance, max_slips) for c in ('input','both')]
        input_ok = all(locks)
        output_ok = is_locked(cases['both']['output_drive'], period, on_min,
                              frequency_tolerance, max_slips)
        off_ok = all(cases[c]['output_drive']['R'] <= off_max for c in ('neither','input','control'))
        inputs.append(input_ok)
        switches.append(input_ok and output_ok and off_ok)
        # This bounded ranking aid rewards a usable input in BOTH control states.
        # It is only a sweep tie-breaker; it never replaces the pass criteria.
        qualities.append(min(cases[c]['input_drive']['R'] /
            (1+abs(cases[c]['input_drive']['frequency_mismatch'])*period+
             cases[c]['input_drive']['slip_count']) for c in ('input','both')))
        contrasts.append(cases['both']['output_drive']['R'] -
                         max(cases[c]['output_drive']['R'] for c in ('neither','input','control')))
    n = len(repeats)
    row = {'valid_repeats':len(valid),'runs':n,
           'input_pass_fraction':sum(inputs)/n,'switch_pass_fraction':sum(switches)/n,
           'input_quality':float(np.mean(qualities)) if valid else None,
           'output_contrast':float(np.mean(contrasts)) if valid else None}
    # Failed phase measurements remain failures and are never replaced with R=0.
    # Means explicitly use only valid repeats; validity counts accompany them.
    for case in CASE_SCALES:
        for relation in ('input_drive','output_drive','output_input'):
            for metric in ('R','frequency_mismatch','slip_count'):
                values = [r['cases'][case][relation][metric] for r in valid]
                key = f'{case}_{relation}_{metric}'
                row[key+'_mean'] = float(np.mean(values)) if values else None
                row[key+'_sem'] = float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
    return row


def run_repeat(base, factor, seed, exe, folder, window, keep):
    """Use the same seed across all four cases and all grid points."""
    cases, errors = {}, {}
    for case in CASE_SCALES:
        config = prepare_config(base, case, factor, seed)
        try:
            metrics, _ = diagnose(config, exe, folder/case, window, keep, save_trace=False)
            cases[case] = metrics
        except ValueError as exc:
            # A strong drive can remove oscillatory crossings. Record that point
            # as unmeasurable instead of mistaking missing phases for good locking.
            errors[case] = str(exc)
    result = {'seed':seed,'status':'invalid_phase' if errors else 'ok','cases':cases,'errors':errors}
    save_json(folder/'repeat.json',result)
    return result


def write_table(out, rows):
    save_json(out/'sweep_summary.json',rows)
    with (out/'sweep_summary.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_grid(out, rows, amplitudes, periods):
    panels = [('input_pass_fraction','Input locked: fraction of repeats',0,1),
              ('both_output_drive_R_mean','Both: output–drive R',0,1),
              ('input_output_drive_R_mean','Input only: output–drive R',0,1),
              ('control_output_drive_R_mean','Control only: output–drive R',0,1),
              ('output_contrast','Both R minus largest off-case R',-1,1),
              ('switch_pass_fraction','All switching checks: fraction of repeats',0,1)]
    fig, axes = plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,(key,title,lo,hi) in zip(axes.flat,panels):
        grid = np.full((len(amplitudes),len(periods)),np.nan)
        for row in rows:
            value = row[key]
            if value is not None:
                grid[amplitudes.index(row['amplitude']), periods.index(row['period'])] = value
        im = ax.imshow(np.ma.masked_invalid(grid),origin='lower',aspect='auto',vmin=lo,vmax=hi,
                       cmap='coolwarm' if lo<0 else 'viridis')
        ax.set_xticks(range(len(periods)),[f'{x:g}' for x in periods],rotation=45)
        ax.set_yticks(range(len(amplitudes)),[f'{x:g}' for x in amplitudes])
        ax.set_xlabel('Input drive period (categorical grid)')
        ax.set_ylabel('Input force amplitude')
        ax.set_title(title)
        fig.colorbar(im,ax=ax)
    fig.savefig(out/'drive_sweep.png',dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--config',type=Path,help='Saved best/result/config JSON; default: bundled base_1_left')
    source.add_argument('--validation-summary',type=Path,help='Select highest old validation score automatically')
    p.add_argument('--amplitudes',nargs='+',type=float,default=[.5,1,2,3,5])
    p.add_argument('--periods',nargs='+',type=float,default=[.14,.15,.16,.17,.184715,.20,.22])
    p.add_argument('--period-mode',choices=['input-only','shared'],default='input-only',
                   help='Keep control period fixed, or explicitly sweep both periods together')
    p.add_argument('--dt-divisor',type=int,choices=[1,2,4],default=4)
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--cores',type=int,default=3)
    p.add_argument('--seed',type=int,default=650000)
    p.add_argument('--window',nargs=2,type=float,default=[60,95])
    p.add_argument('--on-min',type=float,default=.9)
    p.add_argument('--off-max',type=float,default=.3)
    p.add_argument('--frequency-tolerance',type=float,default=.01)
    p.add_argument('--max-slips',type=int,default=0)
    p.add_argument('--exe',type=Path,default=ROOT/'src/rowers')
    p.add_argument('--out',type=Path)
    p.add_argument('--keep',action='store_true')
    args = p.parse_args()
    if min(args.runs,args.cores)<1 or args.seed<0 or args.max_slips<0:
        p.error('Counts must be positive; seed and max-slips nonnegative')
    if not 0<=args.off_max<args.on_min<=1 or not np.isfinite(args.frequency_tolerance) or args.frequency_tolerance<0:
        p.error('Invalid screening thresholds')
    if any(not np.isfinite(x) or x<=0 for x in args.amplitudes+args.periods):
        p.error('Amplitudes and periods must be finite and positive')
    amplitudes, periods = sorted(set(args.amplitudes)), sorted(set(args.periods))
    try:
        path, reason = select_source(args.config,args.validation_summary)
        base = load_configuration(path)
        validate_protocol(base,args.window)
        if args.window[1]-args.window[0] < 2*max(periods):
            raise ValueError('Window must span at least two cycles of every swept period')
        # Preflight every point before launching costly simulations.
        points = [(a,t,driven_config(base,a,t,args.period_mode)) for a in amplitudes for t in periods]
        for _,_,config in points:
            prepare_config(config,'both',args.dt_divisor,args.seed)
    except (ValueError,FileNotFoundError,KeyError) as exc:
        p.error(str(exc))
    if not args.exe.is_file():
        p.error('Missing executable: run make -C src')
    out = (args.out or ROOT/'results'/('drive_sweep_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'selected_geometry.json',{'configuration':base.values,'source':str(path),'selection':reason})
    settings = vars(args).copy()
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in settings.items()}
    settings.update(amplitudes=amplitudes,periods=periods,selected_source=str(path),selection=reason,
                    executable_sha256=fingerprint(args.exe),
                    code_sha256={name:fingerprint(ROOT/'analysis'/name) for name in
                      ('sweep_switch_drive.py','diagnose_switch.py','simulation.py','experiments.py','bead_analysis.py')})
    save_json(out/'protocol.json',settings)
    print(f'{reason}\n{len(points)} grid points, {len(points)*args.runs*4} simulations; output: {out}',flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.cores) as pool:
        for index,(amplitude,period,config) in enumerate(points,1):
            folder = out/f'point_{index:03d}'
            folder.mkdir()
            # Candidate configuration uses the actual refined timestep and sample
            # counters so subsequent validation can reproduce the swept dynamics.
            actual = prepare_config(config,'both',args.dt_divisor,args.seed)
            save_json(folder/'configuration.json',{'configuration':actual.values})
            futures = [pool.submit(run_repeat,config,args.dt_divisor,args.seed+rep,
                       args.exe.resolve(),folder/f'repeat_{rep+1}',args.window,args.keep) for rep in range(args.runs)]
            repeats = [f.result() for f in futures]
            save_json(folder/'replicates.json',repeats)
            row = {'point':folder.name,'amplitude':amplitude,'period':period,
                   **summarise(repeats,period,args.on_min,args.off_max,args.frequency_tolerance,args.max_slips)}
            rows.append(row)
            write_table(out,rows)  # Checkpoint after each completed point.
            print(f'[{index}/{len(points)}] A={amplitude:g}, P={period:g}: input pass {row["input_pass_fraction"]:.2f}, switch pass {row["switch_pass_fraction"]:.2f}, valid {row["valid_repeats"]}/{args.runs}',flush=True)
    complete = [r for r in rows if r['valid_repeats']==args.runs]
    # Never export an invalid phase point as a usable regime.
    ranking = sorted(complete,key=lambda r:(r['input_pass_fraction'],r['input_quality'],r['output_contrast']),reverse=True)
    eligible = [r for r in ranking if r['input_pass_fraction']==1]
    switching = sorted([r for r in eligible if r['switch_pass_fraction']==1],key=lambda r:r['output_contrast'],reverse=True)
    selection = {'status':'No point passed input locking in every repeat' if not eligible else
                 'Provisional input-locking regime found; independent validation required',
                 'best_observed':ranking[0] if ranking else None,
                 'best_input':eligible[0] if eligible else None,
                 'best_switch':switching[0] if switching else None}
    for label in ('best_observed','best_input','best_switch'):
        row = selection[label]
        if row is not None:
            data = json.loads((out/row['point']/'configuration.json').read_text())
            data.update(screening=row,qualification=label,independent_validation_required=True)
            save_json(out/f'{label}.json',data)
    save_json(out/'selection.json',selection)
    plot_grid(out,rows,amplitudes,periods)
    print(f'{selection["status"]}\nSend selection.json and drive_sweep.png from {out}',flush=True)


if __name__ == '__main__':
    main()

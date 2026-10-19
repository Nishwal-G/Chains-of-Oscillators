#!/usr/bin/env python3
"""Controlled quenched stiffness disorder around a saved diode configuration.

Place beside simulation.py, bead_analysis.py and optimisation.py. No optimiser
is run: only Evaluator's forward/backward/unforced experiment is reused.

Example (from repository root):
  python analysis/disorder.py --config results/best.json --exe src/rowers \
      --output results/disorder --strengths 0 .05 .1 .2 --samples 20 --runs 3

w is a maximum fractional perturbation, NOT a standard deviation. Each pattern
u_i ~ Uniform[-1,1] is fixed in time and shared across w: k_i=k_i0*(1+w*u_i).
All baseline geometry, wall settings, individual exponents/excursions and angles,
lambda profiles, signal settings and temperature remain
unchanged unless explicitly overridden. Disorder does not preserve the sample
mean stiffness. A Python seed reproduces disorder, not C++ thermal noise.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import load_configuration, save_json, fingerprint
from optimisation import Evaluator
from bead_analysis import classify_locking


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, type=Path, help='best.json or .input reference')
    p.add_argument('--exe', type=Path, default=Path('src/rowers'))
    p.add_argument('--output', required=True, type=Path, help='New directory; existing paths are refused')
    p.add_argument('--strengths', nargs='+', type=float, default=[0, .025, .05, .1, .15, .2])
    p.add_argument('--samples', type=int, default=20, help='Independent quenched patterns per nonzero w')
    p.add_argument('--runs', type=int, default=3, help='Simulation repeats per fixed configuration')
    p.add_argument('--cores', type=int, default=3, help='Parallel simulation repeats')
    p.add_argument('--seed', type=int, default=12345, help='Disorder seed only')
    p.add_argument('--start', type=float, default=7., help='Analysis start, strictly after signal onset t=5')
    p.add_argument('--total-time', type=float, help='Optional duration override')
    p.add_argument('--temperature', type=float, help='Optional temperature override; 0 for deterministic pilot')
    p.add_argument('--hysteresis', type=float, help='Position-crossing hysteresis; evaluator default if omitted')
    p.add_argument('--slip-hysteresis', type=float, default=np.pi/4)
    p.add_argument('--r-min', type=float, default=.9)
    p.add_argument('--mismatch-max', type=float, default=.01)
    p.add_argument('--slip-rate-max', type=float, default=0.)
    p.add_argument('--keep', action='store_true', help='Keep successful raw trajectories (may be large)')
    p.add_argument('--dry-run', action='store_true', help='Save design/configurations without launching simulator')
    return p


def statistics(values):
    """Across disorder samples, after first averaging thermal repeats.

    SD describes spread of sample means (including residual thermal error).
    SEM describes uncertainty in their mean. Neither is defined from one sample.
    """
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                sd=float(x.std(ddof=1)) if len(x)>1 else None,
                sem=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None)


def sample_summary(record, n, args):
    result = {k: record[k] for k in ('strength', 'sample', 'score', 'forward_R', 'backward_R', 'score_sem')}
    for label, direction, bead in [('forward','forward',n-1), ('backward','backward',0),
                                   ('control_last','control',n-1), ('control_first','control',0)]:
        rows = [next(m for m in rep['metrics'] if m['run']==direction and m['bead']==bead)
                for rep in record['replicates']]
        for key in ('phase_consistency_R', 'relative_frequency_mismatch', 'slip_candidate_rate_per_time'):
            values = [r[key] for r in rows]
            # Absolute mismatch prevents opposite frequency drifts cancelling.
            if key == 'relative_frequency_mismatch':
                values = [abs(v) if v is not None else None for v in values]
                key = 'absolute_relative_frequency_mismatch'
            result[f'{label}_{key}'] = statistics(values)['mean']
        flags = [classify_locking(r, args.r_min, args.mismatch_max, args.slip_rate_max) for r in rows]
        result[f'{label}_locking_fraction'] = None if any(f is None for f in flags) else float(np.mean(flags))
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows):
    groups = []
    for w in sorted({r['strength'] for r in rows}):
        samples = [r for r in rows if r['strength']==w]
        group = {'strength':w, 'n_samples':len(samples)}
        for key in rows[0]:
            if key not in ('strength', 'sample', 'score_sem'):
                for name, value in statistics([r[key] for r in samples]).items():
                    group[f'{key}_{name}'] = value
        groups.append(group)
    return groups


def plots(rows, groups, output):
    """Bands are +/- one disorder SD, not confidence intervals."""
    x = np.array([g['strength'] for g in groups])
    def curve(ax, key, label, color, style='-'):
        y = np.array([g.get(key+'_mean') if g.get(key+'_mean') is not None else np.nan for g in groups])
        sd = np.array([g.get(key+'_sd') if g.get(key+'_sd') is not None else np.nan for g in groups])
        ax.plot(x, y, style, marker='o', label=label, color=color)
        ax.fill_between(x, y-sd, y+sd, color=color, alpha=.13)
    fig, axes = plt.subplots(1, 2, figsize=(11,4.4), constrained_layout=True)
    curve(axes[0], 'forward_R', 'Forward receiver', 'tab:blue')
    curve(axes[0], 'backward_R', 'Backward receiver', 'tab:orange')
    curve(axes[0], 'control_last_phase_consistency_R', 'Unforced last bead', 'tab:blue', '--')
    curve(axes[0], 'control_first_phase_consistency_R', 'Unforced first bead', 'tab:orange', '--')
    axes[0].set_ylabel('Signal phase consistency R')
    for r in rows:
        axes[1].scatter(r['strength'], r['score'], s=12, color='gray', alpha=.45)
    curve(axes[1], 'score', 'Mean signed diode score', 'tab:green')
    axes[1].axhline(0, color='black', lw=.7)
    axes[1].set_ylabel('D = forward R - backward R')
    for ax in axes:
        ax.set_xlabel('Disorder strength w'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Points: sample means; shading: +/- one SD across disorder samples')
    fig.savefig(output/'disorder_response.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15,4.5), constrained_layout=True)
    for label, color in [('forward','tab:blue'), ('backward','tab:orange'),
                         ('control_last','tab:green'), ('control_first','tab:red')]:
        for ax, suffix in zip(axes, ['locking_fraction','absolute_relative_frequency_mismatch',
                                     'slip_candidate_rate_per_time']):
            curve(ax, label+'_'+suffix, label.replace('_',' '), color)
    for ax, ylabel in zip(axes, ['Fraction of repeats locked','Absolute relative frequency mismatch',
                                'Slip candidates per unit time']):
        ax.set(xlabel='Disorder strength w', ylabel=ylabel); ax.grid(alpha=.2)
    axes[0].set_ylim(-.05,1.05); axes[0].legend(fontsize=8)
    fig.suptitle('Finite-window locking diagnostics; unforced comparisons are not transmission')
    fig.savefig(output/'disorder_locking.png', dpi=180); plt.close(fig)


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if min(args.samples,args.runs,args.cores)<1 or args.seed<0:
        p.error('samples, runs and cores must be positive; seed must be nonnegative')
    if any(not np.isfinite(w) or not 0<=w<1 for w in args.strengths):
        p.error('Each disorder strength must be finite and satisfy 0 <= w < 1')
    args.strengths = sorted(set([0., *args.strengths]))
    thresholds = [args.r_min,args.mismatch_max,args.slip_rate_max]
    if not all(np.isfinite(thresholds)) or not 0<=args.r_min<=1 or min(thresholds[1:])<0:
        p.error('Invalid locking thresholds')
    if not np.isfinite(args.start) or args.start<=5:
        p.error('--start must be finite and strictly greater than 5')
    if args.hysteresis is not None and (not np.isfinite(args.hysteresis) or args.hysteresis<0):
        p.error('--hysteresis must be finite and nonnegative')
    if not np.isfinite(args.slip_hysteresis) or not 0<args.slip_hysteresis<np.pi:
        p.error('--slip-hysteresis must lie strictly between 0 and pi')
    base = load_configuration(args.config)
    overrides = {}
    if args.temperature is not None: overrides['temperature'] = args.temperature
    if args.total_time is not None: overrides['total_time'] = args.total_time
    base = base.updated(**overrides)
    if base.values['F0']<=0:
        p.error('Reference F0 must be positive for the directional forcing experiment')
    if base.values['T_signal']<20*base.values['dt']:
        p.error('Need at least 20 integration steps per signal period')
    if base.values['total_time']-args.start<2*base.values['T_signal']+2*base.values['dt']:
        p.error('Not enough observation time; increase --total-time or reduce --start')
    if not args.dry_run and not args.exe.is_file():
        p.error(f'Simulator not found: {args.exe}; build src/rowers first')
    if args.output.exists():
        p.error('Output already exists. Choose a new directory to protect prior results.')
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    patterns = rng.uniform(-1,1,(args.samples,base.n))
    k0 = np.asarray(base.values['kx_per_bead'],float)
    jobs = []
    for wi,w in enumerate(args.strengths):
        for s in ([None] if w==0 else range(args.samples)):
            coefficients = k0.copy() if s is None else k0*(1+w*patterns[s])
            config = base.updated(kx_per_bead=coefficients.tolist())
            name = f'w_{wi:03d}_sample_{s:04d}' if s is not None else 'baseline'
            jobs.append((w,s,name,config))
    design = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  reference_sha256=fingerprint(args.config), reference_configuration=base.values,
                  script_sha256=fingerprint(Path(__file__)), patterns=patterns,
                  planned_configurations=len(jobs), planned_simulations=3*args.runs*len(jobs),
                  note='Python seed controls quenched disorder only; C++ noise is not seeded here.',
                  jobs=[dict(strength=w,sample=s,name=name,configuration=c.values) for w,s,name,c in jobs])
    save_json(args.output/'disorder_setup.json',design)
    print(f'{len(jobs)} configurations; {design["planned_simulations"]} simulations.', flush=True)
    if args.dry_run:
        print(f'Design saved to {args.output}; no simulations launched.')
        return 0
    if base.values['temperature']==0 and args.runs>1:
        print('Note: repeats may be identical at zero temperature with fixed initial conditions.', flush=True)
    evaluator = Evaluator(args.exe,args.output,args.runs,args.cores,args.start,
                          keep=args.keep,all_beads=True,hysteresis=args.hysteresis,
                          slip_hysteresis=args.slip_hysteresis)
    rows = []
    for index,(w,s,name,config) in enumerate(jobs,1):
        print(f'[{index}/{len(jobs)}] w={w:g}, sample={s if s is not None else "baseline"}', flush=True)
        try:
            record = evaluator.evaluate(config)
        except Exception as exc:
            save_json(args.output/'failure.json',dict(strength=w,sample=s,error=str(exc)))
            raise  # Never silently omit a difficult sample from the averages.
        record.update(strength=w,sample=s)
        save_json(args.output/f'{name}.json',record)
        rows.append(sample_summary(record,base.n,args))
        # Checkpoint successful summaries after every configuration.
        write_csv(args.output/'disorder_samples.csv',rows)
        groups = summarise(rows)
        write_csv(args.output/'disorder_summary.csv',groups)
    plots(rows,groups,args.output)
    print(f'Finished. Results and plots: {args.output.resolve()}')
    return 0


if __name__=='__main__':
    sys.exit(main())

"""Plot switching transfer curves, including a diagnostic sigmoid fit.

Examples:
 python analysis/control_nonlinearity_plots.py --results results/control_nonlinearity_fine
 python analysis/control_nonlinearity_plots.py --results results/control_nonlinearity_fine --baseline-results results/control_nonlinearity_initial

No simulations are run by this script. Sigmoid fitting requires scipy.
"""
import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

CASES = {'both': 'Input + control', 'input_only': 'Input only', 'control_only': 'Control only'}


def read_rows(folder):
    with (folder / 'summary.csv').open(newline='') as stream:
        return list(csv.DictReader(stream))


def series(rows, bead, case):
    data = sorted((r for r in rows if r['bead'] == bead and r['case'] == case),
                  key=lambda r: float(r['control_force']))
    x = np.array([float(r['control_force']) for r in data])
    complete = np.array([int(r['n_valid']) == int(r['n_total']) and bool(r['R_mean'])
                         for r in data])
    def metric(key):
        return np.array([float(r[key]) if r[key] else np.nan for r in data])
    return x, complete, {key: metric(key) for key in
                         ('R_mean', 'R_sem', 'mismatch_mean', 'slips_mean')}


def combine_both(primary, baseline, bead):
    # Prefer the fine sweep when both datasets contain the same force.
    point_map = {}
    for source in (baseline, primary):
        for r in source:
            if r['bead'] != bead or r['case'] != 'both':
                continue
            if int(r['n_valid']) != int(r['n_total']) or not r['R_mean']:
                continue
            point_map[round(float(r['control_force']), 10)] = r
    points = [point_map[k] for k in sorted(point_map)]
    x = np.array([float(r['control_force']) for r in points])
    y = np.array([float(r['R_mean']) for r in points])
    sem = np.array([float(r['R_sem']) if r['R_sem'] else np.nan for r in points])
    return x, y, sem


def sigmoid(x, low, high, midpoint, steepness):
    return low + (high - low) * expit(steepness * (x - midpoint))


def fit_sigmoid(x, y, sem):
    if len(x) < 6 or np.ptp(x) <= 0 or np.ptp(y) < 0.025:
        return None, 'Too little amplitude coverage or response variation for an informative fit'
    span = np.ptp(x)
    lo, hi = float(np.min(y)), float(np.max(y))
    # Use SEM for display only. Avoid giving nearly deterministic tiny SEMs huge fit weights.
    x0 = float(np.median(x))
    bounds_lo = [0.0, 0.0, float(np.min(x) - 2 * span), 0.001 / span]
    bounds_hi = [1.0, 1.0, float(np.max(x) + 2 * span), 100.0 / span]
    best = None
    for guess_mid in np.linspace(np.min(x), np.max(x), 9):
        for steep in (1 / span, 4 / span, 12 / span):
            start = [max(0.0, lo - 0.025), min(1.0, hi + 0.025), float(guess_mid), steep]
            result = least_squares(lambda p: sigmoid(x, *p) - y, start,
                bounds=(bounds_lo, bounds_hi), max_nfev=5000)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    p = best.x
    rss = float(np.sum((sigmoid(x, *p) - y) ** 2))
    linear = np.polyfit(x, y, 1)
    linear_rss = float(np.sum((np.polyval(linear, x) - y) ** 2))
    t10 = p[2] - np.log(9) / p[3]
    t90 = p[2] + np.log(9) / p[3]
    # Penalty for extra parameters; AICc only when enough data.
    def aicc(rss, k):
        n = len(x)
        if n <= k + 1:
            return np.nan
        return n * np.log(max(rss / n, 1e-15)) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
    metrics = {'low': float(p[0]), 'high': float(p[1]), 'midpoint': float(p[2]),
               'steepness': float(p[3]), 'F10': float(t10), 'F90': float(t90),
               'width_10_90': float(t90 - t10), 'rss_sigmoid': rss,
               'rss_linear': linear_rss, 'aicc_sigmoid': float(aicc(rss, 4)),
               'aicc_linear': float(aicc(linear_rss, 2)), 'n_points': len(x)}
    flags = []
    if p[1] - p[0] < 0.03:
        flags.append('little fitted contrast')
    if not (np.min(x) < t10 < t90 < np.max(x)):
        flags.append('10–90% transition not fully observed')
    if abs(p[2] - np.min(x)) < 0.05 * span or abs(p[2] - np.max(x)) < 0.05 * span:
        flags.append('midpoint near sampled boundary')
    if metrics['aicc_sigmoid'] >= metrics['aicc_linear']:
        flags.append('sigmoid not preferred over linear by AICc')
    metrics['warning'] = '; '.join(flags) if flags else 'Fit is descriptive, not proof of bistability'
    return metrics, None


def save_fit_figure(folder, bead, x, y, sem, fit):
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    errors = np.where(np.isfinite(sem), sem, 0)
    ax.errorbar(x, y, yerr=errors, fmt='o', markersize=4, capsize=2,
                label='Observed mean ± SEM', color='tab:blue')
    if fit is not None:
        fine = np.linspace(float(np.min(x)), float(np.max(x)), 500)
        p = [fit[k] for k in ('low', 'high', 'midpoint', 'steepness')]
        ax.plot(fine, sigmoid(fine, *p), '-', linewidth=2, color='tab:red',
                label='Four-parameter logistic fit')
        ax.axvline(fit['midpoint'], linestyle='--', color='tab:gray',
                   label=f"Midpoint = {fit['midpoint']:.3g}")
        ax.axhline(fit['low'], linestyle=':', color='tab:gray', alpha=0.6)
        ax.axhline(fit['high'], linestyle=':', color='tab:gray', alpha=0.6)
        ax.text(0.02, 0.03,
                f"10–90% width = {fit['width_10_90']:.3g} force units\n{fit['warning']}",
                transform=ax.transAxes, fontsize=9, va='bottom', wrap=True)
    ax.set(xlabel='Control force amplitude', ylabel='Output input-clock coherence R',
           title=f'{bead} — nonlinear control transfer')
    ax.set_ylim(-0.03, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(loc='best', fontsize=9)
    fig.savefig(folder / f'{bead}_sigmoid_fit.png', dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True,
                        help='Folder containing summary.csv; saved plots go here')
    parser.add_argument('--baseline-results', type=Path,
                        help='Optional coarse sweep with F_control=0 to anchor OFF response')
    args = parser.parse_args()
    root = args.results
    rows = read_rows(root)
    baseline = read_rows(args.baseline_results) if args.baseline_results else []
    if args.baseline_results:
        for key in ('source_hash', 'baseline_control_force', 'baseline_input_force'):
            a = json.loads((root / 'protocol.json').read_text()).get(key)
            b = json.loads((args.baseline_results / 'protocol.json').read_text()).get(key)
            if a != b:
                parser.error(f'Incompatible sweeps: {key} differs ({a} vs {b})')
        prot1 = json.loads((root / 'protocol.json').read_text())
        prot2 = json.loads((args.baseline_results / 'protocol.json').read_text())
        if prot1.get('dt_divisor') != prot2.get('dt_divisor'):
            warnings.warn('Sweeps use different integration timesteps. Compare overlapping results before trusting combined fit.')
        if prot1.get('window') != prot2.get('window'):
            warnings.warn('Sweeps use different observation windows; combined fit may be inconsistent.')
    outputs = sorted({r['bead'] for r in rows if r['bead'].startswith('output_')})
    if not outputs:
        parser.error('No output beads found')
    for bead in ['input'] + outputs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        for case, label in CASES.items():
            x, valid, values = series(rows, bead, case)
            if len(x) == 0:
                continue
            R = values['R_mean'].copy()
            R[~valid] = np.nan
            sem = values['R_sem'].copy()
            sem[~valid] = np.nan
            axes[0, 0].plot(x, R, 'o-', label=label)
            if np.any(np.isfinite(sem)):
                axes[0, 0].fill_between(x, R-sem, R+sem, alpha=0.15)
            for axis, metric in ((axes[1, 0], 'mismatch_mean'),
                                 (axes[1, 1], 'slips_mean')):
                v = values[metric].copy()
                v[~valid] = np.nan
                axis.plot(x, np.abs(v) if metric == 'mismatch_mean' else v,
                          'o-', label=label)
            if case == 'both' and len(x) > 1:
                slopes = np.array([(R[i+1]-R[i])/(x[i+1]-x[i])
                    if valid[i] and valid[i+1] else np.nan for i in range(len(x)-1)])
                axes[0, 1].plot((x[:-1]+x[1:])/2, slopes, 'o-')
        for ax, title, ylabel in zip(axes.flat,
            ['Input-clock coherence', 'Finite-difference susceptibility',
             'Absolute frequency mismatch', 'Phase slips per window'],
            ['R', 'dR/dF_control', '|f_bead - f_input| (cycles/time)',
             'Slip candidates per observation window']):
            ax.set(title=title, xlabel='Control force amplitude', ylabel=ylabel)
            ax.grid(alpha=0.25)
        axes[0, 0].set_ylim(-0.05, 1.05)
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(bead + ' — mean across valid replicates (incomplete points omitted)')
        fig.savefig(root / f'{bead}_transfer.png', dpi=170)
        plt.close(fig)
    fits = []
    for bead in outputs:
        x, y, sem = combine_both(rows, baseline, bead)
        fit, reason = fit_sigmoid(x, y, sem)
        save_fit_figure(root, bead, x, y, sem, fit)
        if fit is None:
            print(f'{bead}: sigmoid not fitted — {reason}')
        else:
            fits.append({'bead': bead, **fit})
            print(f"{bead}: F_mid={fit['midpoint']:.4g}, width10-90={fit['width_10_90']:.4g}; {fit['warning']}")
    with (root / 'sigmoid_fit_summary.json').open('w') as f:
        json.dump(fits, f, indent=2)
    if fits:
        with (root / 'sigmoid_fit_summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(fits[0]))
            writer.writeheader()
            writer.writerows(fits)
    print(f'Saved transfer figures and sigmoid fits for {len(outputs)} outputs in {root}')


if __name__ == '__main__':
    main()

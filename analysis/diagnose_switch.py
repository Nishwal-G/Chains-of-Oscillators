"""Diagnose mutual locking versus drive entrainment, with matched dt and dt/2.

Reuse the simulator and established crossing-phase metrics. This is a fixed
geometry diagnostic, not an optimiser or a replacement switch success score.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation import load_configuration, run_simulation, save_json, fingerprint
from bead_analysis import MultiBeadCoherenceAnalyzer
from experiments import pair_metrics

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('input_drive', 'output_drive', 'output_input')


def prepare_config(base, case, factor, seed):
    """Keep physical sampling/feedback intervals fixed when reducing dt.

    Apply the experiment's 20-samples-per-period cap once at the ORIGINAL dt.
    Multiplying both counters avoids accidentally changing feedback physics.
    """
    if case not in ('input', 'both') or factor not in (1, 2):
        raise ValueError('Expected input/both and timestep divisor 1/2')
    values = base.values
    drives = deepcopy(values['drives'])
    period = min(d['period'] for d in drives if d['waveform'] != 'constant')
    if period < 20 * values['dt']:
        raise ValueError('Drive period needs at least 20 original integration steps')
    write = min(values['sampling_write'], int(period / (20 * values['dt'])))
    if case == 'input':
        drives[1]['amplitude'] = 0
    return base.updated(drives=drives, seed=seed, dt=values['dt']/factor,
                        sampling_write=int(write)*factor,
                        sampling_feedback=int(values['sampling_feedback'])*factor)


def diagnose(config, executable, folder, window, keep):
    """Measure both beads against the input clock and each other on one window."""
    folder.mkdir(parents=True)
    save_json(folder/'configuration.json', {'configuration': config.values})
    raw = folder/'simulation'
    reader = run_simulation(config, executable, raw)
    theta = np.deg2rad(config.values['angles_deg'])
    xy = config.reference_positions
    references = np.column_stack((xy[:,0]*np.cos(theta)+xy[:,1]*np.sin(theta),
                                  -xy[:,0]*np.sin(theta)+xy[:,1]*np.cos(theta)))
    analyser = MultiBeadCoherenceAnalyzer(reader.time_series, reader.analysis_positions, references)
    bead_in, bead_out = config.targets('input')[0], config.targets('output')[0]
    drive = config.values['drives'][0]
    start, stop = window
    t = np.asarray(analyser.time_series)
    mask = (t >= start) & (t <= stop)
    phases = np.column_stack((analyser._get_phase(bead_in), analyser._get_phase(bead_out)))
    # Check the whole window before asking signal metrics to select valid phases.
    if mask.sum() < 3 or not np.all(np.isfinite(phases[mask])):
        raise ValueError('Incomplete crossing-phase coverage; choose an interior window')
    metrics = {}
    for name, bead in [('input_drive', bead_in), ('output_drive', bead_out)]:
        m = analyser.measure_signal_locking(bead, drive['period'], start, stop,
                                           signal_phase=drive['phase'])
        if m['status'] != 'ok':
            raise ValueError(f'{name}: {m["status"]}')
        metrics[name] = {'R': m['phase_consistency_R'],
                         'frequency_mismatch': m['frequency_mismatch_cycles_per_time'],
                         'slip_count': m['slip_candidates_total']}
    metrics['output_input'], _ = pair_metrics(analyser, bead_out, bead_in, start, stop,
                                              drive['period'], np.pi/4)
    clock = 2*np.pi*t/drive['period'] + drive['phase']
    differences = np.column_stack((phases[:,0]-clock, phases[:,1]-clock,
                                    phases[:,1]-phases[:,0])) / (2*np.pi)
    trace = np.column_stack((t[mask], differences[mask]))
    # CSV preserves absolute unwrapped differences. Only plot offsets are removed.
    np.savetxt(folder/'phase_trace.csv.gz', trace, delimiter=',',
               header='time,input_minus_drive_cycles,output_minus_drive_cycles,output_minus_input_cycles', comments='')
    save_json(folder/'metrics.json', metrics)
    if not keep:
        shutil.rmtree(raw)  # Remove only this successful task's bulky raw output.
    return metrics, trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'templates/computing/junction_diagnostic.json')
    p.add_argument('--exe', type=Path, default=ROOT/'src/rowers')
    p.add_argument('--out', type=Path)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--cores', type=int, default=3)
    p.add_argument('--seed', type=int, default=550000)
    p.add_argument('--window', nargs=2, type=float, default=[60,95])
    p.add_argument('--keep', action='store_true', help='Keep large raw simulation files')
    args = p.parse_args()
    base = load_configuration(args.config)
    if min(args.runs, args.cores) < 1 or args.seed < 0:
        p.error('Positive counts and a nonnegative seed required')
    if not args.exe.is_file():
        p.error('Missing executable: run make -C src')
    drives = base.values['drives']
    if not drives or len(drives) != 2 or any(len(base.targets(g)) != 1 for g in ('input','control','output')):
        p.error('Expected two drives and singleton input, control, output groups')
    if any(base.targets(d['targets']) != base.targets(g) for d,g in zip(drives,('input','control'))):
        p.error('Drive 0 must target input; drive 1 must target control')
    if drives[0]['waveform'] == 'constant' or drives[0]['amplitude'] == 0:
        p.error('Input needs a nonzero periodic drive')
    start, stop = args.window
    if not np.all(np.isfinite(args.window)) or not 0 <= start < stop < base.values['total_time']:
        p.error('Choose a finite window inside the simulation, ending before total_time')
    if any(d['start'] > start or d['stop'] < stop for d in drives):
        p.error('Both drive schedules must cover the measurement window')
    if base.values['temperature'] != 0:
        p.error('This matched-initial-condition diagnostic currently requires temperature=0')
    out = (args.out or ROOT/'results'/('switch_diagnostic_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'protocol.json', {'configuration':base.values, 'window':args.window,
              'runs':args.runs,'seed':args.seed,'executable_sha256':fingerprint(args.exe),
              'code_sha256':{name:fingerprint(ROOT/'analysis'/name) for name in
                            ('diagnose_switch.py','simulation.py','experiments.py','bead_analysis.py')}})
    jobs = {}
    with ThreadPoolExecutor(max_workers=args.cores) as pool:
        for rep in range(args.runs):
            for case in ('input','both'):
                for factor in (1,2):
                    config = prepare_config(base, case, factor, args.seed+rep)
                    key = (rep,case,factor)
                    jobs[key] = pool.submit(diagnose, config, args.exe.resolve(),
                        out/f'repeat_{rep+1}'/case/f'dt_div_{factor}', args.window, args.keep)
        results = {}
        for key, future in jobs.items():
            results[key] = future.result()
            print(f'Completed repeat {key[0]+1}, {key[1]}, dt/{key[2]}', flush=True)
    # Plot in the main thread: matplotlib is not thread safe.
    for rep in range(args.runs):
        fig, axes = plt.subplots(3,2,figsize=(12,9),sharex=True)
        for col, case in enumerate(('input','both')):
            for factor, style in ((1,'-'),(2,'--')):
                trace = results[rep,case,factor][1]
                for row in range(3):
                    axes[row,col].plot(trace[:,0], trace[:,row+1]-trace[0,row+1],
                                       style,label=f'dt/{factor}',linewidth=1)
            axes[0,col].set_title('Input only' if case == 'input' else 'Input + control')
            for row, label in enumerate(LABELS):
                axes[row,col].set_ylabel(label.replace('_',' − ')+' (cycles)')
                axes[row,col].grid(alpha=.25)
                axes[row,col].legend()
            axes[2,col].set_xlabel('Simulation time')
        fig.suptitle(f'Repeat {rep+1}: phase drift (initial offset removed; no detrending)')
        fig.tight_layout()
        fig.savefig(out/f'phase_differences_repeat_{rep+1}.png',dpi=150)
        plt.close(fig)
    summary = []
    for case in ('input','both'):
        for relation in LABELS:
            for metric in ('R','frequency_mismatch','slip_count'):
                original = np.array([results[r,case,1][0][relation][metric] for r in range(args.runs)])
                refined = np.array([results[r,case,2][0][relation][metric] for r in range(args.runs)])
                row = {'case':case,'relation':relation,'metric':metric}
                for name, values in [('original',original),('half_dt',refined),('paired_change',refined-original)]:
                    row[name+'_mean'] = float(values.mean())
                    row[name+'_sem'] = float(values.std(ddof=1)/np.sqrt(args.runs)) if args.runs > 1 else None
                summary.append(row)
    save_json(out/'diagnostic_summary.json', {'window':args.window,'runs':args.runs,
              'frequency_units':'cycles per simulation time','comparison':summary})
    print(f'Finished: {out}\nSend diagnostic_summary.json and phase_differences_repeat_1.png',flush=True)


if __name__ == '__main__':
    main()

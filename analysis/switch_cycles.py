"""Repeated-control validation, using the existing continuous-sequence worker."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from simulation import load_configuration, save_json, fingerprint
from signal_switch import classify_locking


def validate_protocol(protocol):
    """Reject ambiguous schedules/thresholds before launching expensive runs."""
    for key in ('cycles', 'seed'):
        if type(protocol[key]) is not int or protocol[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if protocol['amplitude'] not in (2, 3):
        raise ValueError('amplitude must select saved search 2 or 3')
    for key in ('on_duration', 'off_duration', 'settling_time', 'measurement_duration'):
        if not np.isfinite(protocol[key]) or protocol[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if protocol['settling_time'] + protocol['measurement_duration'] >= min(
            protocol['on_duration'], protocol['off_duration']):
        raise ValueError('Measurement must finish before the next transition')
    factors = protocol['dt_divisors']
    if not factors or len(set(factors)) != len(factors) or any(
            type(f) is not int or f < 1 for f in factors):
        raise ValueError('dt_divisors must be distinct positive integers')
    c = protocol['classification']
    if not all(np.isfinite(c[k]) for k in (
            'on_min', 'frequency_tolerance', 'slipping_frequency_min')):
        raise ValueError('Thresholds must be finite')
    if not (0 < c['on_min'] <= 1 and
            0 < c['frequency_tolerance'] < c['slipping_frequency_min']):
        raise ValueError('Require 0 < locked tolerance < slipping threshold')
    if type(c['slipping_min_slips']) is not int or c['slipping_min_slips'] < 1:
        raise ValueError('slipping_min_slips must be a positive integer')


def build_schedule(config, protocol):
    """Five on/off pairs plus a final ON block test recovery after every OFF.

    Drive phases remain referenced to global time; pulses do not reset phase
    or bead state. Measurements exclude the settling part of every block.
    """
    validate_protocol(protocol)
    if len(config.values['drives']) != 2:
        raise ValueError('Saved configuration must contain input and control drives')
    input_drive, control = deepcopy(config.values['drives'])
    if any(d['amplitude'] <= 0 or d['period'] <= 0 for d in (input_drive, control)):
        raise ValueError('Both saved drives must have positive amplitude and period')
    windows, pulses = [], []
    time = max(1., input_drive['start'], control['start'])
    input_drive['start'] = time
    for index in range(2 * protocol['cycles'] + 1):
        on = index % 2 == 0
        duration = protocol['on_duration' if on else 'off_duration']
        stop = time + duration
        start_measure = time + protocol['settling_time']
        windows.append({
            'stage': f'{"on" if on else "off"}_{index // 2 + 1}',
            'control_on': on, 'block': [time, stop],
            'window': [start_measure, start_measure + protocol['measurement_duration']]
        })
        if on:
            pulse = deepcopy(control)
            pulse.update(start=time, stop=stop)
            pulses.append(pulse)
        time = stop
    input_drive['stop'] = time
    return config.updated(total_time=time, drives=[input_drive, *pulses]), windows


def refine_timestep(config, factor):
    """Refine relative to SAVED dt; preserve write/feedback physical intervals."""
    return config.updated(dt=config.values['dt']/factor,
                          sampling_write=config.values['sampling_write']*factor,
                          sampling_feedback=config.values['sampling_feedback']*factor)


def classify_rows(rows, windows, criteria):
    for row, window in zip(rows, windows):
        row.update(control_on=window['control_on'], block=window['block'])
        row['input_state'] = classify_locking(row['input_signal'], criteria)
        states = [classify_locking(m, criteria) for m in row['signal_metrics']]
        row['output_state'] = states[0] if states and len(set(states)) == 1 else 'inconclusive'
        expected = 'locked' if window['control_on'] else 'slipping'
        # Input failure is included in the denominator, never silently discarded.
        row['passed'] = row['input_state'] == 'locked' and row['output_state'] == expected
    return rows


def aggregate(repeats):
    """Cycle windows in one trajectory are correlated: retain repeat-level rates."""
    result = {'repeats': len(repeats),
              'fully_successful_repeats': sum(all(r['passed'] for r in rows) for rows in repeats)}
    for name, on in (('on', True), ('off', False)):
        selected = [[r for r in rows if r['control_on'] == on] for rows in repeats]
        rates = [float(np.mean([r['passed'] for r in rows])) for rows in selected]
        flat = [r for rows in selected for r in rows]
        result[name] = {
            'windows': len(flat), 'passed_windows': sum(r['passed'] for r in flat),
            'fraction': float(np.mean(rates)), 'per_repeat_fractions': rates,
            'mean_R': float(np.mean([r['R'] for r in flat])),
            'input_locked_fraction': float(np.mean([r['input_state']=='locked' for r in flat])),
            'output_inconclusive_windows': sum(r['output_state']=='inconclusive' for r in flat)}
    return result


def plot_trace(t, delta, windows, destination):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout='constrained')
    for column, ax in enumerate(axes):
        valid = np.isfinite(delta[:, column])
        if valid.any():
            ax.plot(t[valid], delta[valid, column] - delta[valid, column][0], lw=1)
        for window in windows:
            if window['control_on']:
                ax.axvspan(*window['block'], color='green', alpha=.12)
        ax.set_ylabel(('Input' if column == 0 else 'Output') + ' − drive (cycles)')
    axes[0].set_title('Green: control ON; phase offset removed, drift retained')
    axes[-1].set_xlabel('Simulation time')
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run_cycles(args, worker):
    """Use frozen best.json, fresh seeds, and identical seeds across dt variants."""
    protocol = json.loads(args.cycle_protocol.read_text())
    validate_protocol(protocol)
    work = args.work.resolve()
    source = work/'search'/f"amplitude_{protocol['amplitude']}"/'best.json'
    base = load_configuration(source)
    if (base.n != 6 or base.targets('input') != [0]
            or base.targets('control') != [2] or base.targets('output') != [5]):
        raise ValueError('Cycle validation expects the saved six-bead switch layout')
    spec = json.loads((work/'experiment.json').read_text())
    if spec.get('objective') != 'signal_switch':
        raise ValueError('Cycle validation requires the signal_switch experiment')
    config, windows = build_schedule(base, protocol)
    destination = args.cycle_out.resolve() if args.cycle_out else work/'cycles'
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination/'protocol.json', protocol)
    save_json(destination/'source_configuration.json', {'configuration': base.values})
    save_json(destination/'experiment.json', spec)
    save_json(destination/'settings.json', {
        'source': str(source), 'source_sha256': fingerprint(source),
        'runs': args.runs, 'cores': args.cores, 'windows': windows,
        'executable_sha256': fingerprint(args.exe.resolve()),
        'code_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
                        ('switch_cycles.py', 'run_signal_switch.py', 'signal_switch.py',
                         'experiments.py', 'simulation.py', 'bead_analysis.py')},
        'note': 'Independent repeats; cycle windows within a repeat are correlated. '
                'Offline crossing phases do not establish causal switching latency.'})
    summary = []
    for factor in protocol['dt_divisors']:
        refined = refine_timestep(config, factor)
        folder = destination/f'dt_{factor}'
        folder.mkdir()
        repeats = []
        with ThreadPoolExecutor(max_workers=args.cores) as pool:
            futures = [pool.submit(worker, refined, spec, folder/f'repeat_{i+1}',
                                   args.exe.resolve(), protocol['seed']+i, windows)
                       for i in range(args.runs)]
            # Consume and save each completed repeat in deterministic order.
            for i, future in enumerate(futures, 1):
                rows, t, delta = future.result()
                classify_rows(rows, windows, protocol['classification'])
                save_json(folder/f'repeat_{i}'/'stages.json', rows)
                plot_trace(t, delta, windows, folder/f'phase_repeat_{i}.png')
                repeats.append(rows)
                save_json(folder/'stage_metrics.json', repeats)
                print(f'dt/{factor}: repeat {i}/{args.runs} saved', flush=True)
        with (folder/'windows.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['repeat','stage','start','stop','input_state','output_state',
                             'passed','R','relative_frequency_mismatch','slips'])
            for i, rows in enumerate(repeats, 1):
                for row in rows:
                    m = row['signal_metrics'][0]
                    writer.writerow([i,row['stage'],*row['window'],row['input_state'],
                                     row['output_state'],row['passed'],row['R'],
                                     m['relative_frequency_mismatch'],m['slip_candidates_total']])
        summary.append({'saved_dt_divisor': factor, 'dt': refined.values['dt'],
                        **aggregate(repeats)})
        save_json(destination/'summary.json', summary)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    for ax, state in zip(axes, ('on','off')):
        for index, item in enumerate(summary):
            rates = item[state]['per_repeat_fractions']
            ax.scatter([index]*len(rates), rates, alpha=.4)
            ax.plot(index, item[state]['fraction'], 'k_', markersize=20)
        ax.set_xticks(range(len(summary)), [f"dt/{s['saved_dt_divisor']}" for s in summary])
        ax.set_ylim(-.05, 1.05)
        ax.set_title('Control '+state.upper())
        ax.set_ylabel('Passing window fraction per repeat')
    fig.savefig(destination/'comparison.png', dpi=150)
    plt.close(fig)
    print(f'Finished: {destination}. Compare dt results before claiming convergence.')

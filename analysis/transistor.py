"""Opt-in multi-amplitude coherence transistor objective; existing physics reused.

R is phase consistency with the input clock, NOT mechanical output power.
All outputs and all repeats are scored before averaging; poor outputs cannot
be hidden by averaging them with good outputs. No sigmoid fit enters search.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from optimisation import Evaluator
from experiments import ExperimentEvaluator
from signal_switch import locking_loss
from simulation import fingerprint


def validate_protocol(p):
    allowed = {'fractions', 'control_max', 'window', 'low_end', 'high_start',
               'on_min', 'off_max', 'frequency_tolerance', 'max_slips', 'weights'}
    if set(p)-allowed:
        raise ValueError(f'Unknown transistor settings: {set(p)-allowed}')
    x = np.asarray(p['fractions'], dtype=float)
    if (x.ndim != 1 or len(x) < 4 or not np.all(np.isfinite(x)) or
            x[0] != 0 or x[-1] != 1 or np.any(np.diff(x) <= 0)):
        raise ValueError('fractions must increase strictly from 0 to 1, with >=4 points')
    a, b = p['low_end'], p['high_start']
    if type(a) is not int or type(b) is not int or not 0 <= a < b < len(x):
        raise ValueError('low_end and high_start must be ordered grid indices')
    if not np.isfinite(p['control_max']) or p['control_max'] <= 0:
        raise ValueError('control_max must be positive')
    w = p['window']
    if len(w) != 2 or not np.all(np.isfinite(w)) or not 0 <= w[0] < w[1]:
        raise ValueError('Invalid observation window')
    if not 0 <= p['off_max'] < p['on_min'] <= 1:
        raise ValueError('Require 0 <= off_max < on_min <= 1')
    if not np.isfinite(p['frequency_tolerance']) or p['frequency_tolerance'] <= 0:
        raise ValueError('Positive frequency_tolerance required')
    if type(p['max_slips']) is not int or p['max_slips'] < 0:
        raise ValueError('Nonnegative integer max_slips required')
    names = {'contrast', 'transition', 'off', 'on', 'leakage', 'monotonicity', 'input'}
    if set(p['weights']) != names or any(not np.isfinite(v) or v <= 0 for v in p['weights'].values()):
        raise ValueError('Supply a positive finite weight for every score component')


def score_transfer(rows, p):
    """rows: one matched repeat, ordered from zero to maximum control force."""
    validate_protocol(p)
    if len(rows) != len(p['fractions']):
        raise ValueError('Transfer grid length differs from protocol')
    beads = [m['bead'] for m in rows[0]['driven']['signal_metrics']]
    if not beads or len(set(beads)) != len(beads):
        raise ValueError('Need unique monitored outputs')
    for row in rows:
        for case in ('driven', 'control_only'):
            metrics = row[case]['signal_metrics']
            if [m['bead'] for m in metrics] != beads:
                raise ValueError('Output ordering must match throughout transfer curve')
            for m in metrics:
                if (m.get('status') != 'ok' or not np.all(np.isfinite([
                        m['phase_consistency_R'], m['relative_frequency_mismatch'],
                        m['slip_candidates_total']])) or not 0 <= m['phase_consistency_R'] <= 1+1e-12):
                    raise ValueError('Invalid output measurement; never treat missing phase as OFF')
        m = row['driven']['input_signal']
        if not np.all(np.isfinite([m['phase_consistency_R'], m['relative_frequency_mismatch'], m['slip_candidates_total']])):
            raise ValueError('Invalid input measurement')
    a, b = p['low_end'], p['high_start']
    input_loss = max(locking_loss(row['driven']['input_signal'], p) for row in rows)
    details = {}
    for j, bead in enumerate(beads):
        r = np.array([v['driven']['signal_metrics'][j]['phase_consistency_R'] for v in rows])
        leakage = np.array([v['control_only']['signal_metrics'][j]['phase_consistency_R'] for v in rows])
        terms = {
            'contrast': float(r[b:].mean()-r[:a+1].mean()),
            # Fixed interval: no max-slope selection which could reward noise.
            'transition': float(r[b]-r[a]),
            'off': float(np.maximum(0, r[:a+1]-p['off_max']).max()),
            'on': max(locking_loss(v['driven']['signal_metrics'][j], p) for v in rows[b:]),
            'leakage': float(np.maximum(0, leakage-p['off_max']).max()),
            'monotonicity': float(np.maximum(0, -np.diff(r)).sum()),
            'input': float(input_loss),
        }
        score = sum(p['weights'][k]*v*(1 if k in ('contrast','transition') else -1)
                    for k,v in terms.items())
        details[str(bead)] = {'score': float(score), 'components': terms,
            'targets_pass': all(terms[k] == 0 for k in ('off','on','leakage','input'))}
    return min(v['score'] for v in details.values()), {
        'outputs': details, 'targets_pass': all(v['targets_pass'] for v in details.values())}


def experiment_spec(config, p):
    return {'kind':'control', 'objective':'signal_switch', 'period':config.values['drives'][0]['period'],
            'on_min':p['on_min'], 'off_max':p['off_max'],
            'frequency_tolerance':p['frequency_tolerance'], 'max_slips':p['max_slips'],
            'cases':[{'name':name, 'scales':scales, 'window':p['window']}
                     for name,scales in [('neither',[0,0]),('input',[1,0]),('control',[0,1]),('both',[1,1])]]}


class TransistorEvaluator(Evaluator):
    def __init__(self, *args, transistor, **kwargs):
        super().__init__(*args, **kwargs)
        validate_protocol(transistor)
        self.transistor = deepcopy(transistor)
        self.protocol.update(objective='coherence_transistor_v1', transistor=self.transistor)
        for name in ('transistor.py','experiments.py','signal_switch.py'):
            self.protocol['code_sha256'][name] = fingerprint(Path(__file__).with_name(name))

    def _replicate(self, config, folder):
        if config.values['seed'] < 0:
            config = config.updated(seed=int(np.random.SeedSequence().generate_state(1)[0]))
        p = self.transistor
        if len(config.values.get('drives') or []) != 2:
            raise ValueError('Need two drives: input then control')
        worker = ExperimentEvaluator(self.executable, folder, runs=1, cores=1,
            keep=True, hysteresis=self.hysteresis, slip_hysteresis=self.slip_hysteresis,
            experiment=experiment_spec(config,p))
        rows = []
        # The original four-case worker is reused without changes. Input-only
        # and neither are deliberately repeated at each nonzero amplitude.
        # This costs 4*(grid_size-1) simulations per repeat; seeds are matched.
        for i, fraction in enumerate(p['fractions'][1:], 1):
            current = config.updated(**{'drives[1].amplitude':fraction*p['control_max']})
            result = worker._replicate(current, folder/f'level_{i}')
            cases = result['cases']
            if i == 1:
                rows.append({'fraction':0., 'amplitude':0., 'driven':cases['input'],
                             'control_only':cases['neither']})
            rows.append({'fraction':fraction, 'amplitude':fraction*p['control_max'],
                         'driven':cases['both'], 'control_only':cases['control']})
        score, details = score_transfer(rows,p)
        return {'seed':config.values['seed'], 'score':score, 'transfer':rows, **details}

    def summarise(self, replicates):
        return {'target_success_fraction':float(np.mean([r['targets_pass'] for r in replicates]))}

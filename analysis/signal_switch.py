"""Opt-in signal-switch score. No dynamics or phase extraction is defined here."""
import numpy as np


def locking_loss(metric, spec):
    """Zero only if coherence, frequency agreement and slip targets all pass.

    Bounded frequency/slip penalties keep one badly drifting bead from
    overwhelming all other requirements. Nonzero losses remain informative.
    """
    if metric.get('status') != 'ok':
        raise ValueError('Signal objective needs complete phase coverage')
    on = spec.get('on_min', .9)
    tolerance = spec.get('frequency_tolerance', .01)
    excess = max(0., abs(metric['relative_frequency_mismatch'])/tolerance-1)
    slips = max(0, metric['slip_candidates_total']-spec.get('max_slips',0))
    return (max(0.,on-metric['phase_consistency_R']) +
            excess/(1+excess) + slips/(1+slips))


def signal_switch_score(rows, spec):
    """Require the input in both states and output only with both drives on.

    Off means low output coherence with the input clock, NOT zero bead motion.
    Assess every output bead and every matched repeat before taking averages.
    """
    off = spec.get('off_max', .3)
    input_losses = [locking_loss(rows[c]['input_signal'],spec) for c in ('input','both')]
    on_losses = [locking_loss(m,spec) for m in rows['both']['signal_metrics']]
    off_losses = {c:[max(0.,m['phase_consistency_R']-off)
                     for m in rows[c]['signal_metrics']] for c in ('neither','input','control')}
    if not on_losses or any(not v for v in off_losses.values()):
        raise ValueError('Signal objective requires output measurements')
    details = {'input_loss':float(np.mean(input_losses)),
               'on_loss':float(np.mean(on_losses)),
               'off_loss':float(np.mean(off_losses['input'])),
               'control_penalty':float(np.mean(off_losses['neither']+off_losses['control'])),
               'input_success':all(v==0 for v in input_losses)}
    details['switch_success'] = (details['input_success'] and all(v==0 for v in on_losses)
                                and all(v==0 for values in off_losses.values() for v in values))
    score = -sum(details[k] for k in ('input_loss','on_loss','off_loss','control_penalty'))
    return float(score),details


def classify_locking(metric, criteria):
    """Diagnostic labels only: leave the existing optimisation score unchanged.

    Slipping requires BOTH sustained frequency drift and detected full slips.
    Intermediate/invalid measurements are inconclusive, never automatic passes.
    """
    keys = ('phase_consistency_R', 'relative_frequency_mismatch',
            'slip_candidates_total')
    if metric.get('status') != 'ok' or any(
            not np.isfinite(metric.get(k, np.nan)) for k in keys):
        return 'inconclusive'
    mismatch = abs(metric['relative_frequency_mismatch'])
    slips = metric['slip_candidates_total']
    if (metric['phase_consistency_R'] >= criteria['on_min']
            and mismatch <= criteria['frequency_tolerance'] and slips == 0):
        return 'locked'
    if (mismatch >= criteria['slipping_frequency_min']
            and slips >= criteria['slipping_min_slips']):
        return 'slipping'
    return 'inconclusive'


# Both objectives use the same external-clock measurements. The original
# signal_switch objective retains its low-coherence OFF target unchanged.
SIGNAL_OBJECTIVES = ('signal_switch', 'signal_slip_switch')


def slipping_loss(metric, spec):
    """Bounded target deficits: do not reward unlimited drift or extra slips."""
    if metric.get('status') != 'ok':
        raise ValueError('Slipping objective needs complete phase coverage')
    drift = abs(metric['relative_frequency_mismatch'])
    slips = metric['slip_candidates_total']
    return (max(0., 1-drift/spec['slipping_frequency_min']) +
            max(0., 1-slips/spec['slipping_min_slips']))


def signal_slip_switch_score(rows, spec):
    """Every monitored output must lock in BOTH and slip in all other cases.

    Penalise the worst output in each case, so adding outputs cannot dilute
    one bad bead's loss. Input locking is required only while input is driven.
    A score of zero meets every finite-window target; negative is worse.
    """
    input_losses = [locking_loss(rows[c]['input_signal'], spec) for c in ('input','both')]
    output = {}
    for case in ('neither','input','control','both'):
        output[case] = {
            str(m['bead']): (locking_loss(m,spec) if case=='both' else slipping_loss(m,spec))
            for m in rows[case]['signal_metrics']}
        if not output[case]:
            raise ValueError('At least one output bead is required')
    bead_ids = set(output['both'])
    if any(set(v)!=bead_ids for v in output.values()):
        raise ValueError('All cases must measure the same outputs')
    details = {
        'input_loss':max(input_losses),
        'on_loss':max(output['both'].values()),
        'off_loss':max(output['input'].values()),
        'control_penalty':max(max(output[c].values()) for c in ('neither','control')),
        'input_success':all(v==0 for v in input_losses),
        'output_success':{b:all(output[c][b]==0 for c in output) for b in bead_ids},
        'output_losses':output,
    }
    details['switch_success'] = details['input_success'] and all(details['output_success'].values())
    return -float(sum(details[k] for k in ('input_loss','on_loss','off_loss','control_penalty'))),details

"""Measurements for transistor characterisation, independent of optimisation.

Existing crossing phases and slip detector are reused. Invalid phase coverage
is explicit; it is never interpreted as an unlocked success.
"""
import numpy as np
from signal_switch import locking_loss


def valid(m):
    return (m.get('status') == 'ok' and all(m.get(k) is not None and np.isfinite(m[k])
        for k in ('phase_consistency_R','relative_frequency_mismatch','slip_candidates_total')))


def locked(m, criteria):
    return valid(m) and (m['phase_consistency_R'] >= criteria['on_min'] and
        abs(m['relative_frequency_mismatch']) <= criteria['frequency_tolerance'] and
        m['slip_candidates_total'] <= criteria['max_slips'])


def loss(m, criteria):
    return float(locking_loss(m,criteria)) if valid(m) else 3.0


def trapezoid(y, x):
    # NumPy 1.24 compatibility, no dependency on a newer integration API.
    return np.sum((y[1:]+y[:-1])*.5*np.diff(x),axis=0)


def bounded_window(t, y, start, stop):
    """Include interpolated endpoints, retaining all interior samples."""
    t=np.asarray(t);y=np.asarray(y)
    if len(t)<3 or not np.all(np.diff(t)>0) or start<t[0] or stop>t[-1] or start>=stop:
        raise ValueError('Window outside increasing sampled trajectory')
    middle=(t>start)&(t<stop)
    xx=np.r_[start,t[middle],stop]
    yy=np.r_[np.interp(start,t,y),y[middle],np.interp(stop,t,y)]
    if not np.all(np.isfinite(yy)): raise ValueError('Nonfinite samples in requested window')
    return xx,yy


def fourier_response(t,q,period,window,phase=0.):
    """2/T integral of projected trap displacement against the input clock.

    Caller uses an integer number of clock periods. No mean subtraction or
    detrending: retain the stated physical observable. Complex response also
    captures phase, unlike its magnitude alone.
    """
    x,y=bounded_window(t,q,*window)
    z=2*trapezoid(y*np.exp(-1j*(2*np.pi*x/period+phase)),x)/(x[-1]-x[0])
    return {'real':float(z.real),'imag':float(z.imag),'amplitude':float(abs(z)),
            'phase_rad':float(np.angle(z)),'window':list(window)}


def phasor(t,phase,period,window,clock_phase=0.):
    x,y=bounded_window(t,phase,*window)
    return trapezoid(np.exp(1j*(y-2*np.pi*x/period-clock_phase)),x)/(x[-1]-x[0])


def metric(analyser, bead, period, window, criteria, clock_phase=0.):
    t=analyser.time_series;phase=analyser._get_phase(bead)
    try:
        bounded_window(t,phase,*window)
    except ValueError as e:
        return {'bead':bead,'status':'invalid_phase_coverage','reason':str(e)}
    m=analyser.measure_signal_locking(bead,period,*window,signal_phase=clock_phase,
                                     slip_hysteresis=criteria['slip_hysteresis'])
    return {'bead':bead,**m}


def phase_response(t0,phi0,t1,phi1,period,pre,post,shift,clock_phase=0.):
    """Paired intervention, measured against the UNCHANGED baseline clock.

    H=(Z_perturbed-Z_baseline)/(exp(i*shift)-1). Perfect tracking gives
    H=Z_baseline, a control-only oscillator gives H=0. Pre-window difference
    is retained as a check, rather than removed to conceal unmatched runs.
    """
    try:
        z0=phasor(t0,phi0,period,post,clock_phase)
        z1=phasor(t1,phi1,period,post,clock_phase)
        pre0=phasor(t0,phi0,period,pre,clock_phase)
        pre1=phasor(t1,phi1,period,pre,clock_phase)
        h=(z1-z0)/np.expm1(1j*shift)
        # Integrate the paired unwrapped phase difference on a common grid.
        x,y=bounded_window(t0,phi0,*post)
        _,_=bounded_window(t1,phi1,*post)
        difference=np.interp(x,t1,phi1)-y
        zdelta=trapezoid(np.exp(1j*difference),x)/(x[-1]-x[0])
        return {'status':'ok','shift_rad':float(shift),'baseline_R':float(abs(z0)),
            'perturbed_R':float(abs(z1)), 'H_real':float(h.real),'H_imag':float(h.imag),
            'response_magnitude':float(abs(h)),
            'tracking_error':float(abs(h-z0)),
            'following_quality':float(max(0.,abs(z0)-abs(h-z0))),
            'phase_change_rad':float(np.angle(zdelta)),
            'phase_change_consistency':float(abs(zdelta)),
            'pre_difference':float(abs(pre1-pre0))}
    except ValueError as e:
        return {'status':'invalid_phase_coverage','reason':str(e)}


def gating_contrast(on,off):
    if not valid(on) or not valid(off):return None
    a,b=on['phase_consistency_R'],off['phase_consistency_R']
    return float((a-b)/(a+b)) if a+b>1e-12 else None


def probability(flags):
    """Wilson interval across independent repeats; None is invalid, not OFF."""
    n=len(flags);values=[bool(x) for x in flags if x is not None]
    m=len(values);k=sum(values)
    result={'total':n,'valid':m,'invalid':n-m,'successes':k,
            'fraction_valid':k/m if m else None,'fraction_all':k/n if n else None,
            'wilson95':None}
    if m:
        z=1.959963984540054;den=1+z*z/m;centre=(k/m+z*z/(2*m))/den
        half=z*np.sqrt((k/m)*(1-k/m)/m+z*z/(4*m*m))/den
        result['wilson95']=[max(0.,centre-half),min(1.,centre+half)]
    return result


def detect_switch(windows, transition, stop, bead, desired_on, criteria, dwell, previous):
    """Apparent window-based delay. Censor misses; never invent a zero time."""
    key=str(bead)
    def state(m):
        if not valid(m):return None
        return locked(m,criteria) if desired_on else m['phase_consistency_R']<=criteria['off_max']
    prior=state(previous)
    if prior is None:return {'status':'invalid_prior_state','delay':None}
    if prior:return {'status':'already_in_target','delay':None}
    selected=[r for r in windows if r['start']>=transition-1e-9 and r['stop']<=stop+1e-9]
    count=0;first=None
    for row in selected:
        if state(row['outputs'][key]) is True and locked(row['input'],criteria):
            count+=1
            if count==1:first=row
            if count>=dwell:
                delay=first['stop']-transition
                return {'status':'detected','delay':float(delay),
                        'confirmed_at':float(row['stop']),
                        'window_width':float(first['stop']-first['start']),
                        'interpretation':'First all-post-transition trailing window of a sustained target run; offline estimate, not causal latency'}
        else:count=0;first=None
    return {'status':'not_reached','delay':None,'observation_limit':float(stop-transition)}

"""Weak-control transistor experiments and opt-in device optimisation.

Schedules use the existing engine's half-open [start,stop) drive segments.
Drive phases always reference absolute simulation time; states never reset.
"""
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
from simulation import run_simulation, save_json, fingerprint
from diagnose_switch import phase_analyser
from optimisation import Evaluator
from transistor_measurements import (metric, valid, locked, loss, fourier_response,
    phase_response, gating_contrast, detect_switch)


def validate_device(p):
    keys={'version','control_ratio','fractions','stationary','phase','cycles','criteria',
          'weights','objective_cycles','training_disorder','robustness','map'}
    if set(p)!=keys or p['version']!=1:raise ValueError('Unknown/missing device protocol fields or version')
    if not np.isfinite(p['control_ratio']) or not 0<p['control_ratio']<1:
        raise ValueError('control_ratio must be strictly between 0 and 1')
    f=np.asarray(p['fractions'],float)
    if f.ndim!=1 or len(f)<3 or not np.all(np.isfinite(f)) or f[0]!=0 or f[-1]!=1 or np.any(np.diff(f)<=0):
        raise ValueError('fractions must increase strictly from 0 to 1 with >=3 points')
    c=p['criteria']
    if set(c)!={'on_min','off_max','frequency_tolerance','max_slips','slip_hysteresis'}:
        raise ValueError('Unexpected criteria fields')
    if (not np.all(np.isfinite(list(c.values()))) or not 0<=c['off_max']<c['on_min']<=1 or
        c['frequency_tolerance']<=0 or type(c['max_slips']) is not int or c['max_slips']<0 or
        not 0<c['slip_hysteresis']<np.pi):raise ValueError('Invalid locking criteria')
    s=p['stationary'];ph=p['phase'];cy=p['cycles']
    if set(s)!={'duration','window'} or set(ph)!={'time','pre','post','shifts'} or set(cy)!={'count','block_duration','window','stride','dwell_windows'}:
        raise ValueError('Unexpected timing fields')
    def window(w):return len(w)==2 and np.all(np.isfinite(w)) and 0<=w[0]<w[1]<s['duration']
    if not np.isfinite(s['duration']) or not window(s['window']) or not window(ph['pre']) or not window(ph['post']):
        raise ValueError('Observation windows must be interior to stationary run')
    if not ph['pre'][1]<ph['time']<ph['post'][0]:raise ValueError('Phase windows must bracket intervention')
    if len(ph['shifts'])<2 or not all(np.isfinite(v) and 0<abs(v)<np.pi for v in ph['shifts']) or not (min(ph['shifts'])<0<max(ph['shifts'])):
        raise ValueError('Test both positive and negative phase shifts smaller than pi')
    if (type(cy['count']) is not int or cy['count']<1 or type(cy['dwell_windows']) is not int or cy['dwell_windows']<1 or
        not np.all(np.isfinite(list(cy.values()))) or not 0<cy['stride']<=cy['window']<cy['block_duration']/2 or
        cy['window']+(cy['dwell_windows']-1)*cy['stride']>=cy['block_duration']):raise ValueError('Invalid cycle settings')
    names={'contrast','difference','phase','off','on','leakage','input','monotonicity','cycles'}
    if set(p['weights'])!=names or any(not np.isfinite(v) or v<0 for v in p['weights'].values()):raise ValueError('Invalid objective weights')
    if type(p['objective_cycles']) is not bool or not np.isfinite(p['training_disorder']) or not 0<=p['training_disorder']<1:
        raise ValueError('Invalid objective cycle/disorder setting')
    if set(p['robustness'])!={'stiffness_fractions','temperature_offsets'} or set(p['map'])!={'frequency_ratios','control_fractions'}:
        raise ValueError('Unexpected robustness/map fields')
    for k,vals in p['robustness'].items():
        if not vals or len(set(vals))!=len(vals) or any(not np.isfinite(v) or v<0 or (k=='stiffness_fractions' and v>=1) for v in vals):raise ValueError('Invalid robustness grid')
    for k,vals in p['map'].items():
        if not vals or len(set(vals))!=len(vals) or any(not np.isfinite(v) or (v<=0 if k=='frequency_ratios' else not 0<=v<=1) for v in vals):raise ValueError('Invalid map grid')


def check_source(config,p):
    validate_device(p)
    drives=config.values.get('drives') or []
    if len(drives)!=2 or len(config.targets('input'))!=1 or len(config.targets('control'))!=1:
        raise ValueError('Need two saved drives and one input and one control bead')
    if config.targets(drives[0]['targets'])!=config.targets('input') or config.targets(drives[1]['targets'])!=config.targets('control'):
        raise ValueError('Drive 0 must target input and drive 1 control')
    outputs=config.targets('output')
    if not outputs or set(outputs)&set(config.targets('input')+config.targets('control')):
        raise ValueError('Outputs must be distinct from driven beads')
    if any(d['waveform']=='constant' for d in drives) or drives[0]['amplitude']<=0:
        raise ValueError('Need periodic drives and positive input amplitude')
    period=drives[0]['period']
    for window in (p['stationary']['window'],p['phase']['pre'],p['phase']['post']):
        if window[1]-window[0]<4*period:raise ValueError('Analysis windows need at least four input periods')
    if p['cycles']['window']<4*period:raise ValueError('Cycle windows need at least four input periods')
    if max(d['start'] for d in drives)>=min(p['stationary']['window'][0],p['phase']['pre'][0]):
        raise ValueError('Drive starts must precede analysis windows')


def actual_control(config,p):return p['control_ratio']*config.values['drives'][0]['amplitude']


def stationary_config(config,p,amplitude,input_on=True,shift=0.):
    check_source(config,p)
    a,b=deepcopy(config.values['drives']);end=p['stationary']['duration']
    a['stop']=b['stop']=end;b['amplitude']=float(amplitude)
    if not input_on:a['amplitude']=0.
    if shift:
        later=deepcopy(a);a['stop']=p['phase']['time'];later['start']=a['stop'];later['phase']+=shift
        drives=[a,b,later]
    else:drives=[a,b]
    return sampled(config.updated(total_time=end,drives=drives))


def sampled(config):
    periods=[d['period'] for d in config.values['drives'] if d['waveform']!='constant']
    # Preserve feedback spacing; cap output spacing for the fastest clock.
    cap=max(1,int(np.floor(min(periods)/(40*config.values['dt']))))
    return config.updated(sampling_write=min(config.values['sampling_write'],cap))


def cycle_config(config,p,count=None):
    count=p['cycles']['count'] if count is None else count
    start=max(d['start'] for d in config.values['drives'])
    duration=p['cycles']['block_duration'];high=actual_control(config,p)
    a,b=deepcopy(config.values['drives']);blocks=[];controls=[]
    end=start+(2*count+1)*duration
    a['stop']=end+2*config.values['drives'][0]['period']
    for i in range(2*count+1):
        lo=start+i*duration;hi=lo+duration;on=bool(i%2)
        blocks.append({'index':i,'start':lo,'stop':hi,'on':on,'amplitude':high if on else 0.})
        segment=deepcopy(b);segment.update(start=lo,stop=hi,amplitude=high if on else 0.)
        controls.append(segment)
    return sampled(config.updated(total_time=a['stop'],drives=[a,*controls])),blocks


def perturb_stiffness(config,fraction,seed):
    rng=np.random.default_rng(seed)
    return config.updated(**{key:(np.asarray(config.values[key])*(1+rng.uniform(-fraction,fraction,config.n))).tolist()
        for key in ('kx_per_bead','ky_per_bead')})


def measure(reader,config,p,window,clock_phase=None):
    analyser=phase_analyser(config,reader)
    d=config.values['drives'][0];period=d['period']
    clock_phase=d['phase'] if clock_phase is None else clock_phase
    inp=config.targets('input')[0];outputs=config.targets('output')
    # Integer-period Fourier window fits within requested observation interval.
    count=int(np.floor((window[1]-window[0])/period+1e-10))
    fw=[window[1]-count*period,window[1]]
    theta=np.deg2rad(config.values['angles_deg']);xy=config.reference_positions
    refs=xy[:,0]*np.cos(theta)+xy[:,1]*np.sin(theta)
    row={'window':list(window),'input':metric(analyser,inp,period,window,p['criteria'],clock_phase),
         'outputs':{},'fourier':{}}
    for bead in [inp,*outputs]:
        if bead!=inp:row['outputs'][str(bead)]=metric(analyser,bead,period,window,p['criteria'],clock_phase)
        q=reader.analysis_positions[:,bead,0]-refs[bead]
        row['fourier'][str(bead)]=fourier_response(reader.time_series,q,period,fw,clock_phase)
    return row,analyser


def simulate(config,exe,folder,keep=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'configuration.json',{'configuration':config.values})
    reader=run_simulation(config,exe,folder/'raw')
    # The reader has loaded all arrays. Failed runs are retained by run_simulation.
    if not keep:shutil.rmtree(folder/'raw')
    return reader


def transfer(config,p,exe,folder,keep=False,fractions=None):
    high=actual_control(config,p);fractions=p['fractions'] if fractions is None else fractions
    rows=[];cache={}
    for i,f in enumerate(fractions):
        row={'fraction':float(f),'amplitude':float(f*high)}
        for input_on,label in ((True,'driven'),(False,'control_only')):
            current=stationary_config(config,p,f*high,input_on)
            r=simulate(current,exe,Path(folder)/f'level_{i}'/label,keep)
            row[label],analyser=measure(r,current,p,p['stationary']['window'])
            if input_on and f in (0,1):
                beads=[config.targets('input')[0],*config.targets('output')]
                cache[float(f)]={'time':np.asarray(r.time_series),
                    'phases':{str(b):analyser._get_phase(b) for b in beads}}
        rows.append(row)
    return rows,cache


def intervention(config,p,exe,folder,cache,keep=False,trace=False):
    result={};beads=[config.targets('input')[0],*config.targets('output')]
    d=config.values['drives'][0]
    for f,name in ((0.,'low'),(1.,'high')):
        base=cache[f];result[name]=[]
        for index,shift in enumerate(p['phase']['shifts']):
            current=stationary_config(config,p,f*actual_control(config,p),shift=shift)
            r=simulate(current,exe,Path(folder)/name/f'shift_{index}',keep)
            analyser=phase_analyser(current,r)
            response={str(b):phase_response(base['time'],base['phases'][str(b)],r.time_series,
                analyser._get_phase(b),d['period'],p['phase']['pre'],p['phase']['post'],shift,d['phase']) for b in beads}
            # Require actual input locking after the intervention as well.
            input_metric=metric(analyser,beads[0],d['period'],p['phase']['post'],p['criteria'],d['phase']+shift)
            output_metrics={str(b):metric(analyser,b,d['period'],p['phase']['post'],p['criteria'],d['phase']+shift) for b in beads[1:]}
            result[name].append({'shift':shift,'response':response,'input':input_metric,'outputs':output_metrics})
            if trace:
                path=Path(folder)/name/f'shift_{index}'/'phase_trace.npz'
                t=np.asarray(r.time_series);stride=max(1,len(t)//6000);select=np.arange(0,len(t),stride)
                delta=np.column_stack([analyser._get_phase(b)-np.interp(t,base['time'],base['phases'][str(b)]) for b in beads])
                np.savez_compressed(path,time=t[select],delta=delta[select],beads=beads,shift=shift,change_time=p['phase']['time'])
    return result


def cycles(config,p,exe,folder,keep=False,count=None,trace=False):
    current,blocks=cycle_config(config,p,count)
    r=simulate(current,exe,folder,keep);analyser=phase_analyser(current,r)
    c=p['cycles'];period=current.values['drives'][0]['period'];phase=current.values['drives'][0]['phase']
    inp=config.targets('input')[0];outputs=config.targets('output');rows=[]
    # Whole-series trailing windows include transitions; detection below uses
    # only windows wholly after a transition. Settled windows are reported too.
    for end in np.arange(blocks[0]['start']+c['window'],blocks[-1]['stop']+1e-9,c['stride']):
        window=[float(end-c['window']),float(end)]
        rows.append({'start':window[0],'stop':window[1],
            'input':metric(analyser,inp,period,window,p['criteria'],phase),
            'outputs':{str(b):metric(analyser,b,period,window,p['criteria'],phase) for b in outputs}})
    settled=[];timings=[]
    for i,block in enumerate(blocks):
        window=[block['stop']-c['window']-period,block['stop']-period]
        row={'block':block,'input':metric(analyser,inp,period,window,p['criteria'],phase),
             'outputs':{str(b):metric(analyser,b,period,window,p['criteria'],phase) for b in outputs}}
        settled.append(row)
        if i:
            timings.append({'block':i,'on':block['on'],'outputs':{str(b):detect_switch(rows,block['start'],block['stop'],b,
                block['on'],p['criteria'],c['dwell_windows'],settled[i-1]['outputs'][str(b)]) for b in outputs}})
    if trace:
        t=np.asarray(r.time_series);select=np.arange(0,len(t),max(1,len(t)//10000));clock=2*np.pi*t/period+phase
        np.savez_compressed(Path(folder)/'phase_trace.npz',time=t[select],beads=[inp,*outputs],
            delta=np.column_stack([(analyser._get_phase(b)-clock)[select] for b in [inp,*outputs]]))
    return {'blocks':blocks,'settled':settled,'windows':rows,'timings':timings}


def score_device(data,p):
    """Worst-output score per repeat; invalid measurements receive finite losses."""
    rows=data['transfer'];criteria=p['criteria'];outputs=rows[0]['driven']['outputs'];details={}
    input_loss=max(loss(r['driven']['input'],criteria) for r in rows)
    for name in ('low','high'):
        for item in data['phase'][name]:
            input_loss=max(input_loss,loss(item['input'],criteria))
            inp=item['response'][data['input_bead']]
            input_loss=max(input_loss,min(3.,inp['tracking_error']) if inp['status']=='ok' else 3.)
    for bead in outputs:
        off=rows[0]['driven']['outputs'][bead];on=rows[-1]['driven']['outputs'][bead]
        good=all(valid(r[case]['outputs'][bead]) for r in rows for case in ('driven','control_only'))
        if good:
            r=np.array([v['driven']['outputs'][bead]['phase_consistency_R'] for v in rows])
            contrast=gating_contrast(on,off)
            terms={'contrast':contrast if contrast is not None else 0.,'difference':float(r[-1]-r[0]),
                'off':max(0.,r[0]-criteria['off_max']), 'on':loss(on,criteria),
                'leakage':max(max(0.,v['control_only']['outputs'][bead]['phase_consistency_R']-criteria['off_max']) for v in rows),
                'monotonicity':float(np.maximum(0,-np.diff(r)).sum())}
        else:terms=dict(contrast=0.,difference=0.,off=3.,on=3.,leakage=3.,monotonicity=1.)
        high=[];low=[]
        for item in data['phase']['high']:
            m=item['response'][bead];high.append(m['following_quality'] if m['status']=='ok' else 0.)
            terms['on']=max(terms['on'],loss(item['outputs'][bead],criteria))
        for item in data['phase']['low']:
            m=item['response'][bead];low.append(min(1.,m['response_magnitude']) if m['status']=='ok' else 1.)
        terms['phase']=min(high)-max(low)
        terms['input']=input_loss;terms['cycles']=0.
        if data.get('cycles') is not None:
            losses=[]
            for row in data['cycles']['settled']:
                m=row['outputs'][bead]
                losses.append(loss(m,criteria) if row['block']['on'] else
                              max(0.,m['phase_consistency_R']-criteria['off_max']) if valid(m) else 3.)
                terms['input']=max(terms['input'],loss(row['input'],criteria))
            # Missed/already-present transitions cost 1, detected delays scale
            # with the configured block duration; delays include window bias.
            delays=[v['outputs'][bead]['delay']/p['cycles']['block_duration']
                    if v['outputs'][bead]['status']=='detected' else 1. for v in data['cycles']['timings']]
            terms['cycles']=max(losses)+float(np.mean(delays))
        score=sum(p['weights'][k]*v*(1 if k in ('contrast','difference','phase') else -1) for k,v in terms.items())
        details[bead]={'score':float(score),'components':terms}
    return min(d['score'] for d in details.values()),details


def device_repeat(config,p,exe,folder,keep=False,with_cycles=True,trace=False,cycle_count=1):
    check_source(config,p)
    rows,cache=transfer(config,p,exe,Path(folder)/'transfer',keep)
    response=intervention(config,p,exe,Path(folder)/'phase',cache,keep,trace)
    dynamic=cycles(config,p,exe,Path(folder)/'cycles',keep,count=cycle_count,trace=trace) if with_cycles else None
    result={'seed':config.values['seed'],'input_bead':str(config.targets('input')[0]),
            'transfer':rows,'phase':response,'cycles':dynamic}
    score,details=score_device(result,p)
    # Search records keep settled states and timings, not every sliding window.
    if dynamic is not None and not trace:dynamic.pop('windows',None)
    return {**result,'score':score,'outputs':details}


class DeviceEvaluator(Evaluator):
    def __init__(self,*args,device,**kwargs):
        super().__init__(*args,**kwargs);validate_device(device);self.device=deepcopy(device)
        self.protocol.update(objective='weak_control_phase_device_v1',device=self.device)
        for name in ('transistor_device.py','transistor_measurements.py','diagnose_switch.py','signal_switch.py'):
            self.protocol['code_sha256'][name]=fingerprint(Path(__file__).with_name(name))
    def _replicate(self,config,folder):
        seed=config.values['seed']
        if seed<0:raise ValueError('Device optimisation requires a fixed simulator seed')
        nominal=device_repeat(config,self.device,self.executable,folder/'nominal',self.keep,self.device['objective_cycles'])
        if self.device['training_disorder']:
            stressed=perturb_stiffness(config,self.device['training_disorder'],seed+100000)
            other=device_repeat(stressed,self.device,self.executable,folder/'disorder',self.keep,self.device['objective_cycles'])
            return {'score':min(nominal['score'],other['score']),'seed':seed,
                    'nominal':nominal,'disorder':other,'disorder_configuration':stressed.values}
        return nominal
    def summarise(self,replicates):return {}

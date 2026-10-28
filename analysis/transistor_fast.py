"""Fast weak-control search: sampled transfer shape, endpoint leakage, locking.

Five input-present trajectories plus neither/control-only = seven per repeat.
Phase interventions, cycles and disorder remain independent validation stages.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from time import perf_counter
from optimisation import Evaluator
from simulation import fingerprint
from transistor_device import check_source,actual_control,stationary_config,simulate,measure
from transistor_measurements import valid,loss

OBJECTIVE='weak_control_transfer_fast_v1'


def validate_fast(spec):
    if set(spec)!={'fractions','low_index','high_index','weights'}:
        raise ValueError('Unknown or missing fast-search setting')
    f=np.asarray(spec['fractions'],float)
    if f.ndim!=1 or len(f)<5 or not np.all(np.isfinite(f)) or f[0]!=0 or f[-1]!=1 or np.any(np.diff(f)<=0):
        raise ValueError('Need >=5 strictly increasing control fractions from zero to one')
    a,b=spec['low_index'],spec['high_index']
    if type(a) is not int or type(b) is not int or not 0<a<b<len(f)-1 or b-a<2:
        raise ValueError('Transition endpoints must be interior, with a sampled point between them')
    names={'contrast','shape','on','leakage','input','monotonicity'}
    if set(spec['weights'])!=names or any(not np.isfinite(v) or v<0 for v in spec['weights'].values()):
        raise ValueError('Invalid fast-search weights')
    if spec['weights']['contrast']<=0 or spec['weights']['shape']<=0:
        raise ValueError('Contrast and transfer-shape weights must be positive')


def fast_score(rows,controls,criteria,spec):
    """Contrast plus excess central rise over the straight endpoint chord.

    Shape=0 for an exactly linear curve, positive for a rise concentrated in
    the specified band. Monotonicity and endpoint locking guard against noise
    and flat/oscillatory curves. This is a design score, not a statistical test.
    """
    validate_fast(spec)
    if len(rows)!=len(spec['fractions']) or len(controls)!=2:
        raise ValueError('Incomplete fast experiment')
    beads=list(rows[0]['outputs'])
    if not beads or any(list(r['outputs'])!=beads for r in [*rows,*controls]):
        raise ValueError('All trajectories must measure the same ordered outputs')
    input_loss=max(loss(r['input'],criteria) for r in rows)
    a,b=spec['low_index'],spec['high_index']
    width=spec['fractions'][b]-spec['fractions'][a]
    details={}
    for bead in beads:
        ms=[r['outputs'][bead] for r in rows]
        leakage=[r['outputs'][bead] for r in controls]
        if all(valid(m) for m in [*ms,*leakage]):
            r=np.array([m['phase_consistency_R'] for m in ms])
            contrast=float(r[-1]-r[0])
            central=float(r[b]-r[a])
            terms={'contrast':contrast,'shape':(central-width*contrast)/(1-width),
                'on':loss(ms[-1],criteria),
                'leakage':max(m['phase_consistency_R'] for m in leakage),
                'monotonicity':float(np.maximum(0,-np.diff(r)).sum())}
        else:
            terms={'contrast':0.,'shape':0.,'on':3.,'leakage':3.,'monotonicity':1.}
        score=sum(spec['weights'][k]*v*(1 if k in ('contrast','shape') else -1) for k,v in terms.items())
        details[bead]={'score_without_input_penalty':float(score),'components':terms}
    score=min(v['score_without_input_penalty'] for v in details.values())-spec['weights']['input']*input_loss
    return float(score),{'outputs':details,'input_loss':float(input_loss)}


class FastDeviceEvaluator(Evaluator):
    def __init__(self,*args,device,fast,**kwargs):
        super().__init__(*args,**kwargs)
        validate_fast(fast)
        self.fast=deepcopy(fast);self.device=deepcopy(device)
        self.device['fractions']=list(fast['fractions'])
        # The saved protocol describes what search actually applies; full
        # validation still explicitly executes cycles and phase interventions.
        self.device['training_disorder']=0.
        self.device['objective_cycles']=False
        from transistor_device import validate_device
        validate_device(self.device)
        self.protocol.update(objective=OBJECTIVE,device=self.device,fast=self.fast,
            simulations_per_repeat=len(fast['fractions'])+2,
            omitted_from_search=['phase_interventions','cycles','stiffness_disorder','intermediate_control_only'])
        for name in ('transistor_fast.py','transistor_device.py','transistor_measurements.py','diagnose_switch.py','signal_switch.py'):
            self.protocol['code_sha256'][name]=fingerprint(Path(__file__).with_name(name))

    def _replicate(self,config,folder):
        check_source(config,self.device)
        if config.values['seed']<0:raise ValueError('Fast search requires a fixed simulator seed')
        high=actual_control(config,self.device)
        rows=[];controls=[]
        for index,f in enumerate(self.fast['fractions']):
            current=stationary_config(config,self.device,f*high,True)
            reader=simulate(current,self.executable,folder/f'input_level_{index}',self.keep)
            row,_=measure(reader,current,self.device,self.device['stationary']['window'])
            rows.append({'fraction':f,'amplitude':f*high,**row})
        for f,label in ((0.,'neither'),(1.,'control_only')):
            current=stationary_config(config,self.device,f*high,False)
            reader=simulate(current,self.executable,folder/label,self.keep)
            row,_=measure(reader,current,self.device,self.device['stationary']['window'])
            controls.append({'fraction':f,'amplitude':f*high,**row})
        score,details=fast_score(rows,controls,self.device['criteria'],self.fast)
        return {'seed':config.values['seed'],'score':score,'input_present':rows,
                'input_absent_endpoints':controls,**details}

    def evaluate(self,config):
        start=perf_counter()
        result=super().evaluate(config)
        print(f'Fast evaluation finished in {perf_counter()-start:.1f} s: score={result["score"]:.5f}',flush=True)
        return result

    def summarise(self,replicates):
        return {'input_loss':float(np.mean([r['input_loss'] for r in replicates]))}

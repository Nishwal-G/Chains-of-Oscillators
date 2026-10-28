"""Prepare geometry search or independently characterise a saved best.json.

Run from repository root. See TRANSISTOR.md for commands and score definition.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation import load_configuration, ParameterSpace, save_json, fingerprint
from transistor import TransistorEvaluator, validate_protocol
from signal_switch import locking_loss
from switch_cycles import refine_timestep


def prepare(args):
    base = load_configuration(args.config)
    if len(base.values.get('drives') or []) != 2:
        raise ValueError('Source needs independent input and control drives')
    if base.values['reference_x'] is None:
        raise ValueError('Source needs explicit reference geometry')
    if args.anchor >= base.n or args.anchor < 0:
        raise ValueError('Anchor bead is outside configuration')
    drives = deepcopy(base.values['drives'])
    if any(d['start'] > args.window[0] for d in drives):
        raise ValueError('Both drives must start before the measurement window')
    for d in drives:
        d['stop'] = args.duration
    base = base.updated(total_time=args.duration, drives=drives, seed=args.seed)
    p = {'fractions':[0.,.25,.45,.55,.75,1.],
         'control_max':args.control_max if args.control_max is not None else drives[1]['amplitude'],
         'window':args.window, 'low_end':2, 'high_start':3,
         'on_min':.9, 'off_max':.3, 'frequency_tolerance':.01, 'max_slips':0,
         'weights':{'contrast':1.,'transition':1.,'off':2.,'on':2.,
                    'leakage':2.,'monotonicity':1.,'input':2.}}
    validate_protocol(p)
    if p['window'][1] >= args.duration or p['window'][1]-p['window'][0] < 2*drives[0]['period']:
        raise ValueError('Window needs >=2 periods and must end before simulation stops')
    # Search every unanchored reference coordinate. Stroke angles, stiffness,
    # wall model, noise, input drive and all original physical settings stay fixed.
    spec = {f'reference_{axis}[{i}]':{'bounds':[float(v-args.half_width),float(v+args.half_width)],'step':args.step}
            for axis in ('x','y') for i,v in enumerate(base.values[f'reference_{axis}']) if i != args.anchor}
    ParameterSpace(base,spec)
    args.out.mkdir(parents=True,exist_ok=False)
    save_json(args.out/'start.json', {'configuration':base.values})
    save_json(args.out/'transistor.json',p)
    save_json(args.out/'search.json',spec)
    save_json(args.out/'preparation.json', {'source':str(args.config.resolve()),
        'source_sha256':fingerprint(args.config), 'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}})
    print(f'Prepared {base.n} beads; {len(spec)} free coordinates; control range 0..{p["control_max"]}')
    print(f'Each candidate costs {4*(len(p["fractions"])-1)} simulations per repeat.')


def sem(x):
    return np.std(x,axis=0,ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.zeros(np.asarray(x).shape[1:])


def sigmoid_diagnostic(x, y):
    """Exploratory bounded fit; no hypothesis test or amplification claim."""
    from scipy.optimize import curve_fit
    from scipy.special import expit
    def model(v, lo, hi, centre, width):
        return lo+(hi-lo)*expit((v-centre)/width)
    try:
        fit, _ = curve_fit(model,x,y,p0=[max(0.,min(.2,y[0])),min(1.,max(.8,y[-1])),x[-1]/2,x[-1]/10],
            bounds=([0,0,0,x[-1]*.001],[1,1,x[-1],x[-1]*10]), maxfev=20000)
        predicted = model(x,*fit)
        linear = np.polyval(np.polyfit(x,y,1),x)
        return {'status':'ok', 'parameters':dict(zip(('low','high','centre','width'),map(float,fit))),
                'sigmoid_rmse':float(np.sqrt(np.mean((predicted-y)**2))),
                'linear_rmse':float(np.sqrt(np.mean((linear-y)**2))),
                'note':'Exploratory fit of mean R; four parameters versus two for linear. Lower training RMSE is not evidence of nonlinearity.',
                'predicted':predicted.tolist()}
    except (ValueError, RuntimeError, FloatingPointError) as e:
        return {'status':'failed','error':str(e)}


def report(record, p, out, fit_sigmoid=False):
    repeats = record['replicates']
    rows = repeats[0]['transfer']
    x = np.array([r['amplitude'] for r in rows])
    raw = []
    for k,rep in enumerate(repeats):
        for row in rep['transfer']:
            for case in ('driven','control_only'):
                for m in row[case]['signal_metrics']:
                    raw.append({'repeat':k,'seed':rep['seed'],'control_amplitude':row['amplitude'],
                        'case':case,'bead':m['bead'],'R':m['phase_consistency_R'],
                        'relative_frequency_mismatch':m['relative_frequency_mismatch'],
                        'slips':m['slip_candidates_total'],'locked':locking_loss(m,p)==0})
    with (out/'per_repeat.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0])); w.writeheader(); w.writerows(raw)
    summary, fits = [], {}
    for j,metric in enumerate(rows[0]['driven']['signal_metrics']):
        bead = metric['bead']
        r=np.array([[v['driven']['signal_metrics'][j]['phase_consistency_R'] for v in rep['transfer']] for rep in repeats])
        leak=np.array([[v['control_only']['signal_metrics'][j]['phase_consistency_R'] for v in rep['transfer']] for rep in repeats])
        mismatch=np.array([[v['driven']['signal_metrics'][j]['relative_frequency_mismatch'] for v in rep['transfer']] for rep in repeats])
        slips=np.array([[v['driven']['signal_metrics'][j]['slip_candidates_total'] for v in rep['transfer']] for rep in repeats])
        locked=np.array([[locking_loss(v['driven']['signal_metrics'][j],p)==0 for v in rep['transfer']] for rep in repeats])
        # Adjacent differences of each matched repeat preserve seed covariance.
        slope=np.diff(r,axis=1)/np.diff(x)
        fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        ax=axes[0,0]
        ax.errorbar(x,r.mean(0),yerr=sem(r),fmt='o-',label='Input + control')
        ax.errorbar(x,leak.mean(0),yerr=sem(leak),fmt='s--',label='Control only')
        ax.axhline(p['on_min'],color='green',ls=':',label='ON target')
        ax.axhline(p['off_max'],color='red',ls=':',label='OFF target')
        ax.set(ylim=(-.03,1.03),ylabel='Output-to-input-clock R',title=f'Output bead {bead}')
        if fit_sigmoid:
            fit=sigmoid_diagnostic(x,r.mean(0));fits[str(bead)]=fit
            if fit['status']=='ok': ax.plot(x,fit['predicted'],':',label='Exploratory sigmoid fit')
        ax.legend(fontsize=8)
        axes[0,1].errorbar((x[:-1]+x[1:])/2,slope.mean(0),yerr=sem(slope),fmt='o-')
        axes[0,1].axhline(0,color='k',lw=.6)
        axes[0,1].set(ylabel='Finite-difference dR/dF_control',title='Response slope (not power gain)')
        axes[1,0].errorbar(x,mismatch.mean(0),yerr=sem(mismatch),fmt='o-')
        for s in (-1,1): axes[1,0].axhline(s*p['frequency_tolerance'],color='grey',ls=':')
        axes[1,0].set(ylabel='Relative frequency mismatch')
        axes[1,1].plot(x,locked.mean(0),'o-',label='Output locked fraction')
        axes[1,1].set(ylabel='Fraction of repeats',ylim=(-.03,1.03));axes[1,1].legend()
        for ax in axes.flat: ax.set_xlabel('Control force amplitude');ax.grid(alpha=.2)
        fig.savefig(out/f'output_{bead}_transfer.png',dpi=170);plt.close(fig)
        for i,amplitude in enumerate(x):
            summary.append({'bead':bead,'control_amplitude':float(amplitude),
                'R_mean':float(r[:,i].mean()),'R_sem':float(sem(r)[i]),
                'control_only_R_mean':float(leak[:,i].mean()),
                'frequency_mismatch_mean':float(mismatch[:,i].mean()),
                'slips_mean':float(slips[:,i].mean()),'locked_fraction':float(locked[:,i].mean())})
    input_r=np.array([[v['driven']['input_signal']['phase_consistency_R'] for v in rep['transfer']] for rep in repeats])
    input_ok=np.array([[locking_loss(v['driven']['input_signal'],p)==0 for v in rep['transfer']] for rep in repeats])
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.errorbar(x,input_r.mean(0),yerr=sem(input_r),fmt='o-',label='Input R')
    ax.plot(x,input_ok.mean(0),'s--',label='Input locked fraction')
    ax.set(xlabel='Control force amplitude',ylabel='Input integrity',ylim=(-.03,1.03));ax.legend()
    fig.savefig(out/'input_integrity.png',dpi=170);plt.close(fig)
    save_json(out/'input_summary.json',{'amplitudes':x.tolist(),'R_mean':input_r.mean(0),
        'R_sem':sem(input_r),'locked_fraction':input_ok.mean(0)})
    save_json(out/'summary.json',summary)
    save_json(out/'sigmoid_diagnostics.json',fits)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)


def analyse(args):
    base=load_configuration(args.config)
    source=json.loads(args.config.read_text()) if args.config.suffix=='.json' else {}
    p=json.loads(args.protocol.read_text()) if args.protocol else source.get('protocol',{}).get('transistor')
    if p is None: raise ValueError('Old best.json requires --protocol path/to/transistor.json')
    validate_protocol(p)
    if base.values['seed'] >= 0:
        training=set(range(base.values['seed'],base.values['seed']+source.get('protocol',{}).get('runs',1)))
        if training.intersection(range(args.seed,args.seed+args.runs)):
            raise ValueError('Validation seeds overlap search seeds; choose another --seed')
    # Preserve the exact training grid and its endpoints in the dense sweep.
    dense=deepcopy(p)
    low,high=p['fractions'][p['low_end']],p['fractions'][p['high_start']]
    dense['fractions']=sorted(p['fractions'] + [float(v) for v in np.linspace(0,1,args.points)
        if not np.any(np.isclose(v,p['fractions'],rtol=0,atol=1e-12))])
    dense['low_end']=dense['fractions'].index(low);dense['high_start']=dense['fractions'].index(high)
    if not args.exe.is_file(): raise ValueError('Build src/rowers first')
    args.out.mkdir(parents=True,exist_ok=False)
    save_json(args.out/'source.json',source or {'configuration':base.values})
    save_json(args.out/'analysis_settings.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    comparisons=[]
    for factor in args.dt_divisors:
        config=refine_timestep(base.updated(seed=args.seed),factor)
        for name,protocol in [('validation',p),('dense',dense)]:
            folder=args.out/f'dt_{factor}'/name
            evaluator=TransistorEvaluator(args.exe,folder,runs=args.runs,cores=args.cores,
                                           keep=args.keep,transistor=protocol)
            record=evaluator.evaluate(config)
            save_json(folder/'result.json',record)
            report(record,protocol,folder,args.fit_sigmoid and name=='dense')
            if name=='validation':
                comparisons.append({'dt_divisor':factor,'dt':config.values['dt'],
                    'score':record['score'],'score_sem':record['score_sem'],
                    'target_success_fraction':record['target_success_fraction']})
            print(f'Completed dt/{factor} {name}: {folder}',flush=True)
        save_json(args.out/'validation_comparison.json',comparisons)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','analyse'],required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--exe',type=Path,default=Path('src/rowers'))
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--duration',type=float,default=100.)
    parser.add_argument('--window',type=float,nargs=2,default=[60.,95.])
    parser.add_argument('--control-max',type=float)
    parser.add_argument('--anchor',type=int,default=4)
    parser.add_argument('--half-width',type=float,default=6.)
    parser.add_argument('--step',type=float,default=.5)
    parser.add_argument('--runs',type=int,default=20)
    parser.add_argument('--cores',type=int,default=3)
    parser.add_argument('--points',type=int,default=21)
    parser.add_argument('--dt-divisors',type=int,nargs='+',default=[1,2])
    parser.add_argument('--seed',type=int,help='Default: 5250000 prepare, 8250000 analyse')
    parser.add_argument('--fit-sigmoid',action='store_true')
    parser.add_argument('--keep',action='store_true')
    args=parser.parse_args()
    if args.seed is None: args.seed=5250000 if args.stage=='prepare' else 8250000
    if (args.runs<1 or args.cores<1 or args.points<4 or args.seed<0 or args.seed+args.runs>=2**32 or
        not np.all(np.isfinite([args.duration,args.half_width,args.step])) or
        min(args.duration,args.half_width,args.step)<=0 or
        any(f<1 for f in args.dt_divisors) or len(set(args.dt_divisors))!=len(args.dt_divisors)):
        parser.error('Invalid counts, seed, geometry settings or timestep divisors')
    {'prepare':prepare,'analyse':analyse}[args.stage](args)

if __name__=='__main__':
    main()

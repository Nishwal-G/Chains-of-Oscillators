"""Per-output reports; repeats, not time samples, are the uncertainty unit."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation import save_json
from transistor_measurements import valid, locked, gating_contrast, probability


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def mean_sem(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return (float(a.mean()) if len(a) else None,
            float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None,len(a))


def stat(values):
    m,s,n=mean_sem(values)
    return {'mean':m,'sem':s,'valid_repeats':n}


def measured(m,key):return m[key] if valid(m) else np.nan


def plot_error(ax,x,values,label):
    means=[];errors=[]
    for v in values:
        m,s,_=mean_sem(v);means.append(np.nan if m is None else m);errors.append(0 if s is None else s)
    ax.errorbar(x,means,yerr=errors,fmt='o-',label=label)


def transfer_report(repeats,p,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=repeats[0]['transfer'];outputs=list(rows[0]['driven']['outputs'])
    x=np.array([r['amplitude'] for r in rows]);raw=[];summary=[];contrasts=[];sensitivity=[]
    for j,row in enumerate(rows):
        for case in ('driven','control_only'):
            flags=[]
            for rep in repeats:
                current=rep['transfer'][j][case]
                ms=list(current['outputs'].values())
                flags.append(all(locked(m,p['criteria']) for m in ms) if all(valid(m) for m in ms) else None)
            summary.append({'amplitude':float(x[j]),'case':case,'bead':'joint',**probability(flags)})
    for bead in outputs:
        rv=[];cv=[];av=[];frequencies=[];prob=[]
        for j,row in enumerate(rows):
            for k,rep in enumerate(repeats):
                for case in ('driven','control_only'):
                    current=rep['transfer'][j][case];m=current['outputs'][bead];fourier=current['fourier'][bead]
                    raw.append({'repeat':k,'seed':rep['seed'],'bead':bead,'amplitude':x[j], 'case':case,
                        'status':m['status'],'R':m.get('phase_consistency_R'),
                        'relative_frequency_mismatch':m.get('relative_frequency_mismatch'),
                        'slips':m.get('slip_candidates_total'),'locked':locked(m,p['criteria']) if valid(m) else None,
                        'A':fourier['amplitude'],'Z_real':fourier['real'],'Z_imag':fourier['imag']})
            ms=[r['transfer'][j]['driven']['outputs'][bead] for r in repeats]
            cm=[r['transfer'][j]['control_only']['outputs'][bead] for r in repeats]
            rv.append([measured(m,'phase_consistency_R') for m in ms]);cv.append([measured(m,'phase_consistency_R') for m in cm])
            av.append([r['transfer'][j]['driven']['fourier'][bead]['amplitude'] for r in repeats])
            frequencies.append([measured(m,'relative_frequency_mismatch') for m in ms])
            result=probability([locked(m,p['criteria']) if valid(m) else None for m in ms]);prob.append(result)
            summary.append({'amplitude':float(x[j]),'case':'driven','bead':bead,**result})
            # Same data, several operational criteria; no re-simulation.
            for threshold in (.9,.95,.98):
                for tolerance in (.005,.01,.02):
                    criteria={**p['criteria'],'on_min':threshold,'frequency_tolerance':tolerance}
                    sensitivity.append({'bead':bead,'amplitude':float(x[j]),'R_threshold':threshold,
                        'frequency_tolerance':tolerance,**probability([locked(m,criteria) if valid(m) else None for m in ms])})
        values=[gating_contrast(r['transfer'][-1]['driven']['outputs'][bead],r['transfer'][0]['driven']['outputs'][bead]) for r in repeats]
        if len(x)>1 and rows[0]['fraction']==0 and rows[-1]['fraction']==1:
            contrasts.append({'bead':bead,'per_repeat':values,**stat([v if v is not None else np.nan for v in values])})
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        plot_error(axes[0,0],x,rv,'Input + control');plot_error(axes[0,0],x,cv,'Control only')
        axes[0,0].set(ylabel='R to input clock',ylim=(-.03,1.03));axes[0,0].legend()
        means=[v['fraction_valid'] if v['fraction_valid'] is not None else np.nan for v in prob]
        axes[0,1].plot(x,means,'o-',label='P(locked | valid)')
        axes[0,1].plot(x,[v['fraction_all'] for v in prob],'s--',label='Locked / all repeats')
        for a,v in zip(x,prob):
            if v['wilson95']:axes[0,1].plot([a,a],v['wilson95'],color='grey')
        axes[0,1].set(ylabel='Locking probability',ylim=(-.03,1.03));axes[0,1].legend(fontsize=8)
        plot_error(axes[0,2],x,frequencies,'Output');axes[0,2].set_ylabel('Relative frequency mismatch')
        plot_error(axes[1,0],x,av,'Input + control');axes[1,0].set_ylabel('Fourier displacement amplitude A')
        if len(x)>1:
            slopeR=np.diff(np.asarray(rv),axis=0)/np.diff(x)[:,None]
            slopeA=np.diff(np.asarray(av),axis=0)/np.diff(x)[:,None]
            mid=(x[:-1]+x[1:])/2
            plot_error(axes[1,1],mid,slopeR,'Coherence sensitivity');axes[1,1].set_ylabel('Delta R / Delta control force')
            plot_error(axes[1,2],mid,slopeA,'Displacement response');axes[1,2].set_ylabel('Delta A / Delta control force')
            write_csv(out/f'output_{bead}_slopes.csv',[{'control_midpoint':float(v),'dR_dF':stat(slopeR[i])['mean'],
                'dR_dF_sem':stat(slopeR[i])['sem'],'dA_dF':stat(slopeA[i])['mean'],'dA_dF_sem':stat(slopeA[i])['sem']} for i,v in enumerate(mid)])
        for ax in axes.flat:ax.set_xlabel('Control force amplitude');ax.grid(alpha=.2)
        fig.suptitle(f'Output bead {bead}; means: repeat SEM; probabilities: Wilson 95% intervals')
        fig.savefig(out/f'output_{bead}_transfer.png',dpi=155);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');input_summary=[]
    input_values=[]
    for j in range(len(x)):
        ms=[r['transfer'][j]['driven']['input'] for r in repeats]
        input_values.append([measured(m,'phase_consistency_R') for m in ms])
        input_summary.append({'amplitude':float(x[j]),**probability([locked(m,p['criteria']) if valid(m) else None for m in ms])})
    plot_error(ax,x,input_values,'Input R');ax.plot(x,[v['fraction_all'] for v in input_summary],'s--',label='Input locked / all repeats')
    ax.set(xlabel='Control force amplitude',ylabel='Input integrity',ylim=(-.03,1.03));ax.legend()
    fig.savefig(out/'input_integrity.png',dpi=150);plt.close(fig)
    write_csv(out/'per_repeat.csv',raw);save_json(out/'locking_probabilities.json',summary)
    save_json(out/'contrast.json',contrasts);save_json(out/'threshold_sensitivity.json',sensitivity)
    save_json(out/'input_integrity.json',input_summary)
    return {'contrast':contrasts,'locking':summary}


def phase_report(repeats,p,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);table=[]
    beads=list(repeats[0]['phase']['high'][0]['response'])
    for bead in beads:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for mode in ('low','high'):
            changes=[];quality=[]
            for j,shift in enumerate(p['phase']['shifts']):
                metrics=[r['phase'][mode][j]['response'][bead] for r in repeats]
                # Unwrap each circular response about its expected target, not
                # about zero; prevents +/-pi wrap cancellation in averaging.
                target=shift if mode=='high' or bead==repeats[0]['input_bead'] else 0.
                ch=[target+np.angle(np.exp(1j*(m['phase_change_rad']-target))) if m['status']=='ok' else np.nan for m in metrics]
                q=[m['following_quality'] if m['status']=='ok' else np.nan for m in metrics]
                changes.append(ch);quality.append(q)
                for k,m in enumerate(metrics):table.append({'repeat':k,'bead':bead,'control':mode,'shift':shift,
                    'status':m['status'],'phase_change_rad':m.get('phase_change_rad'),
                    'response_magnitude':m.get('response_magnitude'),'tracking_error':m.get('tracking_error'),
                    'following_quality':m.get('following_quality'),'pre_difference':m.get('pre_difference')})
            plot_error(axes[0],p['phase']['shifts'],changes,mode+' control')
            plot_error(axes[1],p['phase']['shifts'],quality,mode+' control')
        axes[0].plot(p['phase']['shifts'],p['phase']['shifts'],'k:',label='Unit phase following')
        axes[0].set(ylabel='Paired output phase change [rad]');axes[1].set(ylabel='Following quality Q',ylim=(-.03,1.03))
        for ax in axes:ax.set_xlabel('Applied input phase shift [rad]');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle(f'Bead {bead}: input phase changed, control phase held fixed')
        fig.savefig(out/f'bead_{bead}_phase_response.png',dpi=150);plt.close(fig)
    write_csv(out/'phase_response.csv',table)


def cycle_report(repeats,p,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);timings=[];settled=[];summaries=[]
    for k,rep in enumerate(repeats):
        result=rep['cycles'];outputs=list(result['settled'][0]['outputs'])
        for row in result['timings']:
            for b,m in row['outputs'].items():timings.append({'repeat':k,'bead':b,'block':row['block'],'on':row['on'],
                'status':m['status'],'apparent_delay':m['delay']})
        for row in result['settled']:
            for b,m in row['outputs'].items():settled.append({'repeat':k,'bead':b,'block':row['block']['index'],
                'on':row['block']['on'],'status':m['status'],'R':m.get('phase_consistency_R'),
                'relative_frequency_mismatch':m.get('relative_frequency_mismatch'),'slips':m.get('slip_candidates_total')})
        # One example trace per batch; all repeats retain quantitative windows.
        if k==0:
            windows=result['windows'];t=[r['stop'] for r in windows]
            fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
            for block in result['blocks']:
                axes[0].plot([block['start'],block['stop']],[block['amplitude']]*2,'k-')
            for b in outputs:
                axes[1].plot(t,[measured(r['outputs'][b],'phase_consistency_R') for r in windows],label=f'Bead {b}')
                axes[2].plot(t,[measured(r['outputs'][b],'relative_frequency_mismatch') for r in windows],label=f'Bead {b}')
            axes[0].set_ylabel('Control envelope');axes[1].set_ylabel('Trailing-window R');axes[2].set_ylabel('Relative frequency mismatch')
            axes[2].set_xlabel('Simulation time');axes[1].legend();axes[2].legend()
            fig.savefig(out/'cycles_repeat_0.png',dpi=155);plt.close(fig)
    # Aggregate repeated windows within each repeat first. No binomial CI over
    # correlated cycles; use independent-repeat all-cycle success instead.
    all_flags=[]
    for b in outputs:
        flags=[]
        for rep in repeats:
            good=[];missing=False
            for row in rep['cycles']['settled']:
                m=row['outputs'][b];im=row['input']
                if not valid(m) or not valid(im):missing=True
                good.append(valid(m) and locked(im,p['criteria']) and
                    (locked(m,p['criteria']) if row['block']['on'] else m['phase_consistency_R']<=p['criteria']['off_max']))
            flags.append(None if missing else all(good))
        all_flags.append(flags)
        summaries.append({'bead':b,**probability(flags)})
    joint=[None if any(v is None for v in values) else all(values) for values in zip(*all_flags)]
    summaries.append({'bead':'joint',**probability(joint)})
    write_csv(out/'switching_times.csv',timings);write_csv(out/'settled_windows.csv',settled)
    save_json(out/'cycle_reliability.json',summaries)


def trace_plots(root,p):
    root=Path(root)
    for path in root.glob('repeat_0/phase/*/shift_*/phase_trace.npz'):
        a=np.load(path);fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        for j,b in enumerate(a['beads']):ax.plot(a['time'],a['delta'][:,j],label=f'Bead {b}')
        ax.axvline(float(a['change_time']),color='k',ls=':');ax.axhline(float(a['shift']),color='grey',ls='--')
        ax.set(xlabel='Time',ylabel='Perturbed minus baseline phase [rad]');ax.legend()
        fig.savefig(path.with_suffix('.png'),dpi=150);plt.close(fig)
    path=root/'repeat_0/cycles/phase_trace.npz'
    if path.exists():
        a=np.load(path);fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        for j,b in enumerate(a['beads']):ax.plot(a['time'],a['delta'][:,j]/(2*np.pi),label=f'Bead {b}')
        ax.set(xlabel='Time',ylabel='Unwrapped phase minus input clock [cycles]');ax.legend()
        fig.savefig(path.with_suffix('.png'),dpi=150);plt.close(fig)

"""Physics-independent objective regressions and mocked orchestration tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from transistor import score_transfer, TransistorEvaluator, validate_protocol
from simulation import load_configuration, save_json
from run_transistor import report
from optimisation import main

P={'fractions':[0,.25,.45,.55,.75,1.], 'control_max':2., 'window':[60,95],
   'low_end':2, 'high_start':3, 'on_min':.9, 'off_max':.3,
   'frequency_tolerance':.01,'max_slips':0,
   'weights':dict(contrast=1.,transition=1.,off=2.,on=2.,leakage=2.,monotonicity=1.,input=2.)}

def metric(r, bead=5, drift=0., slips=0):
    return dict(bead=bead,status='ok',phase_consistency_R=r,
                relative_frequency_mismatch=drift,slip_candidates_total=slips)

def curve(values, leakage=.05):
    return [dict(amplitude=f*P['control_max'],fraction=f,
        driven=dict(signal_metrics=[metric(v)],input_signal=metric(.99,0)),
        control_only=dict(signal_metrics=[metric(leakage)])) for f,v in zip(P['fractions'],values)]

class TransistorTests(unittest.TestCase):
    def test_prefers_sharp_gated_curve(self):
        ideal=score_transfer(curve([.05,.05,.05,.98,.98,.98]),P)[0]
        for values in ([.5]*6,[.05]*6,[.98]*6,[.05,.25,.45,.55,.75,.98], [.05,.8,.05,.98,.5,.98]):
            self.assertGreater(ideal,score_transfer(curve(values),P)[0])
        self.assertGreater(ideal,score_transfer(curve([.05,.05,.05,.98,.98,.98],.98),P)[0])

    def test_frequency_slips_input_and_worst_output(self):
        rows=curve([.05,.05,.05,.98,.98,.98]);good=score_transfer(rows,P)[0]
        for key,value in [('relative_frequency_mismatch',.1),('slip_candidates_total',3)]:
            bad=deepcopy(rows);bad[4]['driven']['signal_metrics'][0][key]=value
            self.assertLess(score_transfer(bad,P)[0],good)
        bad=deepcopy(rows);bad[3]['driven']['input_signal']=metric(.2,0)
        self.assertLess(score_transfer(bad,P)[0],good)
        for row in rows:
            row['driven']['signal_metrics'].append(metric(.1,6))
            row['control_only']['signal_metrics'].append(metric(.05,6))
        self.assertLess(score_transfer(rows,P)[0],good)
        self.assertFalse(score_transfer(rows,P)[1]['targets_pass'])

    def test_invalid_data_not_off(self):
        rows=curve([.05]*6);rows[0]['driven']['signal_metrics'][0]['phase_consistency_R']=float('nan')
        with self.assertRaises(ValueError):score_transfer(rows,P)
        for change in ({'fractions':[0,.5,.5,1]}, {'control_max':0}, {'low_end':4,'high_start':3}):
            with self.assertRaises(ValueError):validate_protocol({**P,**change})

    def test_evaluate_search_checkpoint_and_reports(self):
        base=load_configuration(ROOT/'templates/computing/junction.json')
        drives=deepcopy(base.values['drives'])
        for d in drives:d['stop']=100
        base=base.updated(total_time=100,drives=drives,seed=1000)
        seen=[]
        def worker(self,config,folder):
            folder.mkdir(parents=True,exist_ok=True)
            seen.append(config.values['seed'])
            high=config.values['drives'][1]['amplitude']>=1.
            def row(r):return dict(signal_metrics=[metric(r)],input_signal=metric(.99,0))
            return {'cases':{'input':row(.05),'neither':row(.05),'control':row(.05),'both':row(.98 if high else .05)}}
        with tempfile.TemporaryDirectory() as tmp, patch('transistor.ExperimentEvaluator._replicate',worker):
            root=Path(tmp);exe=root/'fake_binary';exe.write_text('mock executable fingerprint')
            evaluator=TransistorEvaluator(exe,root/'evaluation',runs=2,cores=2,transistor=P)
            result=evaluator.evaluate(base)
            self.assertEqual(result['target_success_fraction'],1.)
            self.assertEqual(set(seen),{1000,1001})
            self.assertEqual(len(seen),10)
            self.assertIsNone(result['work_directory'])
            report(result,P,root/'evaluation',True)
            self.assertTrue((root/'evaluation/output_5_transfer.png').is_file())
            save_json(root/'config.json',{'configuration':base.values});save_json(root/'protocol.json',P)
            save_json(root/'search.json',{'reference_x[0]':{'bounds':[-26,-22],'step':.1}})
            argv=['optimisation.py','--config',str(root/'config.json'),'--search',str(root/'search.json'),
                  '--transistor',str(root/'protocol.json'),'--exe',str(exe),'--out',str(root/'search'),
                  '--iterations','1','--runs','2','--seed','42']
            with patch.object(sys,'argv',argv):main()
            best=json.loads((root/'search/best.json').read_text())
            self.assertEqual(best['protocol']['objective'],'coherence_transistor_v1')
            self.assertEqual(load_configuration(root/'search/best.json').n,6)
            self.assertTrue((root/'search/progress.png').is_file())
            from run_transistor import main as analyse_main
            with patch.object(sys,'argv',['run_transistor.py','--stage','analyse',
                    '--config',str(root/'search/best.json'),'--exe',str(exe),
                    '--out',str(root/'validation'),'--runs','2','--points','4',
                    '--dt-divisors','1','2','--seed','2000']):
                analyse_main()
            comparison=json.loads((root/'validation/validation_comparison.json').read_text())
            self.assertEqual(len(comparison),2)
            self.assertEqual(comparison[1]['dt'],comparison[0]['dt']/2)
            self.assertTrue((root/'validation/dt_2/dense/summary.csv').is_file())

if __name__=='__main__':unittest.main()

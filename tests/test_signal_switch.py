"""Signal scoring must not confuse mutual coherence with external entrainment."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from experiments import control_score,validate_experiment
from signal_switch import locking_loss

class SignalSwitchTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads((ROOT/'templates/computing/control_signal_switch.json').read_text())
        good={'status':'ok','phase_consistency_R':.98,'relative_frequency_mismatch':0.,'slip_candidates_total':0}
        self.rows={c:{'R':.99,'input_signal':deepcopy(good),'signal_metrics':[deepcopy(good)]}
                   for c in ('neither','input','control','both')}
        for c in ('neither','input','control'):
            self.rows[c]['signal_metrics'][0]['phase_consistency_R']=.1
    def test_targets_and_leakage(self):
        validate_experiment(self.spec)
        score,details=control_score(self.rows,self.spec)
        self.assertEqual(score,0)
        self.assertTrue(details['switch_success'])
        self.rows['input']['signal_metrics'][0]['phase_consistency_R']=.95
        score,details=control_score(self.rows,self.spec)
        self.assertLess(score,0)
        self.assertFalse(details['switch_success'])
    def test_high_R_cannot_hide_slips_or_detuning(self):
        for key,value in [('slip_candidates_total',1),('relative_frequency_mismatch',.02)]:
            rows=deepcopy(self.rows)
            rows['both']['signal_metrics'][0][key]=value
            score,details=control_score(rows,self.spec)
            self.assertLess(score,0)
            self.assertFalse(details['switch_success'])
        self.rows['both']['input_signal']['phase_consistency_R']=.5
        _,details=control_score(self.rows,self.spec)
        self.assertFalse(details['input_success'])
    def test_invalid_coverage_and_thresholds(self):
        m=self.rows['input']['input_signal']; m['status']='insufficient_phase_coverage'
        with self.assertRaises(ValueError): locking_loss(m,self.spec)
        self.spec['frequency_tolerance']=0
        with self.assertRaises(ValueError): validate_experiment(self.spec)
    def test_old_objective_is_unchanged(self):
        rows={c:{'R':v} for c,v in zip(('neither','input','control','both'),(.2,.2,.2,.95))}
        score,details=control_score(rows,{'objective':'switch','off_max':.3,'on_min':.9})
        self.assertEqual(score,0)
        self.assertTrue(details['switch_success'])

class PipelinePreparationTests(unittest.TestCase):
    def test_validation_preserves_saved_dt_and_perturbs_only_geometry(self):
        """Exercise validation preparation without rerunning the costly engine."""
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        import run_signal_switch as runner
        from simulation import load_configuration,save_json
        from diagnose_switch import prepare_config
        base=load_configuration(ROOT/'templates/computing/junction_diagnostic.json')
        saved=prepare_config(base,'both',4,950000)
        spec=json.loads((ROOT/'templates/computing/control_signal_switch.json').read_text())
        search=json.loads((ROOT/'templates/computing/search_geometry_offset.json').read_text())
        calls=[]
        def record(config,specification,out,exe,runs,cores):
            calls.append((config,specification))
            return {'score':-1}
        with TemporaryDirectory() as temp:
            work=Path(temp)
            save_json(work/'base.json',{'configuration':base.values})
            save_json(work/'experiment.json',spec); save_json(work/'search.json',search)
            for amplitude in (2,3):
                folder=work/'search'/f'amplitude_{amplitude}';folder.mkdir(parents=True)
                save_json(folder/'best.json',{'configuration':saved.values})
            with patch.object(sys,'argv',['runner','--stage','validate','--work',str(work),
                                          '--runs','1','--perturbations','1','--exe',sys.executable]),patch.object(runner,'evaluate',record):
                runner.main()
        self.assertEqual(len(calls),6)
        for start in (0,3):
            a,b,jitter=[calls[start+i][0] for i in range(3)]
            self.assertEqual(a.values['dt'],saved.values['dt'])
            self.assertEqual(b.values['dt'],saved.values['dt']/2)
            self.assertEqual(a.values['dt']*a.values['sampling_feedback'],
                             b.values['dt']*b.values['sampling_feedback'])
            self.assertEqual(jitter.values['kx_per_bead'],a.values['kx_per_bead'])
            self.assertEqual(jitter.values['reference_x'][4],0)
            self.assertNotEqual(jitter.values['reference_x'],a.values['reference_x'])
            self.assertEqual(calls[start][1]['cases'][0]['window'],[120,190])

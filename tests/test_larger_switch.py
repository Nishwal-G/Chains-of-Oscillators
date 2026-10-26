"""Larger-array preparation and worst-output locking/slipping regression tests."""
import json
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from simulation import load_configuration, ParameterSpace
from larger_switch import prepare_arrays
from experiments import validate_experiment, control_score, ExperimentEvaluator
from signal_switch import slipping_loss, locking_loss
from switch_load import append_active_rower


class LargerSwitchTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads((ROOT/'templates/computing/control_slip_switch.json').read_text())
        self.base=load_configuration(ROOT/'templates/computing/junction_diagnostic.json')
        self.geometry=json.loads((ROOT/'templates/computing/larger_geometry.json').read_text())
        self.good=dict(status='ok',phase_consistency_R=.97,
                       relative_frequency_mismatch=.001,slip_candidates_total=0)
        self.slip=dict(status='ok',phase_consistency_R=.52,
                       relative_frequency_mismatch=.06,slip_candidates_total=4)

    def rows(self):
        return {case:dict(input_signal=deepcopy(self.good),
                    signal_metrics=[dict(bead=b,**deepcopy(self.good if case=='both' else self.slip))
                                    for b in (5,6)],R=.9)
                for case in ('neither','input','control','both')}

    def test_new_score_and_no_low_coherence_requirement(self):
        validate_experiment(self.spec)
        score,d=control_score(self.rows(),self.spec)
        self.assertEqual(score,0)
        self.assertTrue(d['switch_success'])
        self.assertEqual(slipping_loss(self.slip,self.spec),0)
        # No reward for making already-sufficient drift arbitrarily large.
        self.assertEqual(slipping_loss({**self.slip,'relative_frequency_mismatch':1.},self.spec),0)

    def test_every_output_and_every_case_required(self):
        for case in ('neither','input','control','both'):
            rows=self.rows()
            rows[case]['signal_metrics'][1]=dict(bead=6,**(self.slip if case=='both' else self.good))
            score,d=control_score(rows,self.spec)
            self.assertLess(score,0)
            self.assertFalse(d['switch_success'])
            # Adding another good bead cannot dilute the existing failure.
            for name in rows:
                rows[name]['signal_metrics'].append(dict(bead=7,**(self.good if name=='both' else self.slip)))
            more,_=control_score(rows,self.spec)
            self.assertEqual(score,more)
        rows=self.rows();rows['input']['input_signal']=self.slip
        self.assertFalse(control_score(rows,self.spec)[1]['switch_success'])

    def test_progressive_deficits_and_thresholds(self):
        low={**self.slip,'relative_frequency_mismatch':0.,'slip_candidates_total':0}
        medium={**low,'relative_frequency_mismatch':.01,'slip_candidates_total':1}
        self.assertGreater(slipping_loss(low,self.spec),slipping_loss(medium,self.spec))
        self.assertGreater(slipping_loss(medium,self.spec),0)
        for key,value in [('slipping_frequency_min',.005),('slipping_min_slips',0),
                          ('max_slips',1),('off_max',.3)]:
            spec={**self.spec,key:value}
            with self.assertRaises(ValueError):validate_experiment(spec)

    def test_seven_and_eight_bead_geometry_only(self):
        original=deepcopy(self.base.values)
        for count in (7,8):
            starts,search=prepare_arrays(self.base,count,self.geometry)
            self.assertEqual(len(starts),3)
            self.assertEqual(len(search),7+2*(count-6))
            self.assertTrue(all(k.startswith(('reference_x[','reference_y[')) for k in search))
            for c in starts:
                self.assertEqual(c.n,count)
                self.assertEqual(c.targets('output'),list(range(5,count)))
                self.assertEqual(c.values['dt'],self.base.values['dt'])
                self.assertEqual(c.reference_positions[4].tolist(),self.base.reference_positions[4].tolist())
                self.assertEqual(c.values['kx_per_bead'][:6],original['kx_per_bead'])
                self.assertEqual([c.targets(d['targets']) for d in c.values['drives']],[[0],[2]])
                ParameterSpace(c,search)
        self.assertEqual(self.base.values,original)

    def test_summary_reports_input_and_joint_success(self):
        evaluator=object.__new__(ExperimentEvaluator)
        evaluator.experiment=self.spec
        _,good=control_score(self.rows(),self.spec)
        rows=self.rows();rows['both']['signal_metrics'][1]=dict(bead=6,**self.slip)
        _,bad=control_score(rows,self.spec)
        result=evaluator.summarise([good,bad])
        self.assertEqual(result['switch_successes'],1)
        self.assertEqual(result['input_success_fraction'],1)


if __name__=='__main__':unittest.main()

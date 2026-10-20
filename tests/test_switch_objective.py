"""Target scoring regressions; these tests do not claim a working device."""
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from experiments import control_score, validate_experiment, ExperimentEvaluator
from simulation import load_configuration, ParameterSpace

def rows(off,on,control=.2):
    return {k:{'R':v} for k,v in zip(('input','both','neither','control'),(off,on,.2,control))}

class SwitchTests(unittest.TestCase):
    def test_score(self):
        spec={'objective':'switch','off_max':.3,'on_min':.9}
        score,metrics=control_score(rows(.3,.9),spec)
        self.assertEqual(score,0)
        self.assertTrue(metrics['switch_success'])
        self.assertAlmostEqual(control_score(rows(.8,.85),spec)[0],-.55)
        self.assertAlmostEqual(control_score(rows(.2,.95,.6),spec)[0],-.4)
        self.assertFalse(control_score(rows(.31,1),spec)[1]['switch_success'])
        self.assertAlmostEqual(control_score(rows(.6,.9,.4),{})[0],.1)

    def test_repeat_success(self):
        evaluator=object.__new__(ExperimentEvaluator)
        evaluator.experiment={'objective':'switch'}
        # Mean bars meet both targets but neither individual repeat does.
        reps=[control_score(rows(.1,.8),evaluator.experiment)[1],
              control_score(rows(.5,1),evaluator.experiment)[1]]
        result=evaluator.summarise(reps)
        self.assertEqual(result['switch_success_fraction'],0)
        self.assertEqual(result['switch_trials'],2)

    def test_templates(self):
        folder=ROOT/'templates/computing'
        for name in ('control_switch_long','control_switch_validation','control'):
            validate_experiment(json.loads((folder/f'{name}.json').read_text()))
        spec=json.loads((folder/'control_switch_long.json').read_text())
        for updates in ({'off_max':.95},{'on_min':float('nan')},{'objective':'typo'},{'objective':'contrast'}):
            with self.assertRaises(ValueError):validate_experiment({**spec,**updates})
        config=load_configuration(folder/'junction_switch_long.json')
        space=ParameterSpace(config,json.loads((folder/'search_switch.json').read_text()))
        self.assertEqual(len(space.initial),4)
        self.assertEqual(space.configuration(space.initial).values['kx_per_bead'],[10]*6)

if __name__=='__main__':unittest.main()

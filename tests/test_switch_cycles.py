"""Physical schedule, timestep and conservative classification checks."""
import json
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
from simulation import load_configuration
from signal_switch import classify_locking
from switch_cycles import build_schedule, refine_timestep, validate_protocol, classify_rows, aggregate


class CycleTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads((ROOT/'templates/computing/switch_cycles.json').read_text())
        self.config = load_configuration(ROOT/'templates/computing/junction_diagnostic.json')

    def test_schedule(self):
        config, windows = build_schedule(self.config, self.p)
        self.assertEqual(len(windows), 11)
        self.assertEqual(len(config.values['drives']), 7)
        self.assertEqual(config.values['total_time'], 716)
        self.assertEqual(windows[0]['window'], [46,61])
        for left, right in zip(windows, windows[1:]):
            self.assertEqual(left['block'][1], right['block'][0])
        for row in windows:
            self.assertGreater(row['window'][0],row['block'][0])
            self.assertLess(row['window'][1],row['block'][1])
        for pulse in config.values['drives'][1:]:
            self.assertEqual(pulse['phase'],self.config.values['drives'][1]['phase'])
        self.assertEqual(config.values['reference_x'],self.config.values['reference_x'])

    def test_physical_sampling(self):
        config = refine_timestep(self.config,2)
        for key in ('sampling_write','sampling_feedback'):
            self.assertEqual(config.values['dt']*config.values[key],
                             self.config.values['dt']*self.config.values[key])
        self.assertEqual(config.values['drives'],self.config.values['drives'])

    def test_classification_and_input_failure(self):
        good = dict(status='ok',phase_consistency_R=.97,
                    relative_frequency_mismatch=0.,slip_candidates_total=0)
        slip = dict(status='ok',phase_consistency_R=.5,
                    relative_frequency_mismatch=.06,slip_candidates_total=4)
        c=self.p['classification']
        self.assertEqual(classify_locking(good,c),'locked')
        self.assertEqual(classify_locking(slip,c),'slipping')
        ambiguous={**slip,'relative_frequency_mismatch':.005}
        self.assertEqual(classify_locking(ambiguous,c),'inconclusive')
        self.assertEqual(classify_locking({**good,'status':'invalid'},c),'inconclusive')
        _,windows=build_schedule(self.config,self.p)
        rows=[dict(R=.97 if w['control_on'] else .5,input_signal=deepcopy(good),
                   signal_metrics=[deepcopy(good if w['control_on'] else slip)])
              for w in windows]
        rows[1]['input_signal']['status']='invalid'
        classify_rows(rows,windows,c)
        result=aggregate([rows])
        self.assertEqual(result['fully_successful_repeats'],0)
        self.assertEqual(result['on']['fraction'],1)
        self.assertEqual(result['off']['fraction'],.8)
        self.assertEqual(result['off']['windows'],5)

    def test_invalid_protocol(self):
        for key,value in [('cycles',0),('dt_divisors',[1,1]),('settling_time',60)]:
            p=deepcopy(self.p);p[key]=value
            with self.assertRaises(ValueError): validate_protocol(p)

if __name__=='__main__': unittest.main()

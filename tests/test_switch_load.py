"""Regression tests for a genuinely unforced load and per-bead outcomes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from simulation import load_configuration, PER_BEAD_DEFAULTS, Configuration
from switch_load import add_load, summarise_repeats
from switch_cycles import build_schedule


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.base=load_configuration(ROOT/'templates/computing/junction_diagnostic.json')
        self.p=json.loads((ROOT/'templates/computing/switch_load.json').read_text())
        self.p['amplitude']=3

    def test_original_beads_unchanged_load_is_unforced(self):
        before=deepcopy(self.base.values)
        loaded=add_load(self.base,12)
        self.assertEqual(self.base.values,before)
        self.assertEqual(loaded.n,7)
        for key in (*PER_BEAD_DEFAULTS,'angles_deg','reference_x','reference_y'):
            self.assertEqual(loaded.values[key][:6],self.base.values[key])
        for key in (*PER_BEAD_DEFAULTS,'angles_deg'):
            self.assertEqual(loaded.values[key][6],loaded.values[key][5])
        self.assertAlmostEqual(loaded.values['reference_x'][6]-loaded.values['reference_x'][5],12)
        self.assertEqual(loaded.targets('output'),[5,6])
        for drive in loaded.values['drives']:
            self.assertNotIn(6,loaded.targets(drive['targets']))
        # Input writer/reader must retain N=7, all arrays, groups and forcing.
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'load.input'; loaded.write(path)
            restored=Configuration.read(path)
            self.assertEqual(restored.n,7)
            self.assertEqual(restored.targets('output'),[5,6])
            self.assertEqual(restored.values['drives'],loaded.values['drives'])

    def test_rotated_load_direction_and_invalid_gap(self):
        rotated=self.base.updated(**{'angles_deg[5]':90})
        loaded=add_load(rotated,12)
        self.assertAlmostEqual(loaded.values['reference_x'][6],loaded.values['reference_x'][5])
        self.assertAlmostEqual(loaded.values['reference_y'][6]-loaded.values['reference_y'][5],12)
        for gap in (-1,0,1,float('nan')):
            with self.assertRaises(ValueError): add_load(self.base,gap)

    def test_load_failure_cannot_be_hidden_by_original_output(self):
        _,windows=build_schedule(add_load(self.base,12),self.p)
        good=dict(status='ok',phase_consistency_R=.99,
                  relative_frequency_mismatch=0.,slip_candidates_total=0)
        slip=dict(status='ok',phase_consistency_R=.5,
                  relative_frequency_mismatch=.06,slip_candidates_total=4)
        rows=[]
        for w in windows:
            output={**(good if w['control_on'] else slip),'bead':5}
            load={**slip,'bead':6}  # Load never locks.
            rows.append(dict(stage=w['stage'],window=w['window'],R=.7,
                             input_signal=good,signal_metrics=[output,load]))
        result=summarise_repeats([rows],windows,self.p['classification'],[5,6])
        self.assertEqual(result['beads']['5']['on']['fraction'],1)
        self.assertEqual(result['beads']['6']['on']['fraction'],0)
        self.assertEqual(result['joint']['on']['fraction'],0)
        self.assertEqual(result['joint']['off']['fraction'],1)


if __name__=='__main__': unittest.main()

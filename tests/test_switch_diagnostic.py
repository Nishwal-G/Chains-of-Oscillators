"""Check that timestep refinement does not change the physical protocol."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from diagnose_switch import prepare_config
from simulation import load_configuration

class DiagnosticTests(unittest.TestCase):
    def test_matched_protocol(self):
        base = load_configuration(ROOT/'templates/computing/junction_diagnostic.json')
        original = prepare_config(base,'both',1,550000)
        refined = prepare_config(base,'both',2,550000)
        for key in ('sampling_write','sampling_feedback'):
            self.assertEqual(original.values['dt']*original.values[key],
                             refined.values['dt']*refined.values[key])
        for key in base.values:
            if key not in ('dt','sampling_write','sampling_feedback'):
                self.assertEqual(original.values[key],refined.values[key])
        off = prepare_config(base,'input',1,550000)
        self.assertEqual(off.values['drives'][0],base.values['drives'][0])
        self.assertEqual(off.values['drives'][1]['amplitude'],0)
        self.assertEqual(base.values['drives'][1]['amplitude'],5)
        for key in ('reference_x','reference_y','kx_per_bead','alpha_per_bead'):
            self.assertEqual(off.values[key],base.values[key])

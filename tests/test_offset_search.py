"""Check geometry preparation without duplicating simulator/optimiser tests."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from run_offset_search import prepare_starts, TEMPLATES
from simulation import load_configuration, ParameterSpace

class OffsetTests(unittest.TestCase):
    def test_preparation_preserves_physics(self):
        source=TEMPLATES/'junction_switch_long.json'
        spec=json.loads((TEMPLATES/'search_geometry_offset.json').read_text())
        base=load_configuration(source)
        starts=prepare_starts([source,source],spec)
        self.assertEqual(len(starts),6)
        for label,path,config in starts:
            for key in ('kx_per_bead','alpha_per_bead','angles_deg','mobility','dt'):
                self.assertEqual(config.values[key],base.values[key])
            self.assertEqual([d['amplitude'] for d in config.values['drives']],[.5,5])
            self.assertEqual(config.values['reference_x'][4],0)
            space=ParameterSpace(config,spec)
            self.assertEqual(len(space.initial),7)
            # Test extremes where same-arm separations are smallest.
            for vector in (space.lo,space.hi):
                c=space.configuration(vector)
                self.assertGreaterEqual(np.min(c.pair_distances()),5)

    def test_standalone_templates(self):
        spec=json.loads((TEMPLATES/'search_geometry_offset.json').read_text())
        for label in ('centre','left','right'):
            config=load_configuration(TEMPLATES/f'junction_offset_{label}.json')
            ParameterSpace(config,spec)

if __name__=='__main__': unittest.main()

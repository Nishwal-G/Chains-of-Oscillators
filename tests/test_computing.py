"""Configuration and physical integration regressions; run from project root.

These tests check implementation, not the existence of a working logic device.
The force test compares matched initial states, so its expected result follows
from the applied forces rather than from long-time synchronisation assumptions.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
from simulation import Configuration, ParameterSpace, load_configuration, run_simulation
from experiments import validate_experiment, pair_metrics
from bead_analysis import MultiBeadCoherenceAnalyzer


class ComputingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_configuration(ROOT/'templates/computing/junction.json')

    def test_old_templates_and_roundtrip(self):
        for path in (ROOT/'templates').glob('*.input'):
            old = Configuration.read(path)
            self.assertIsNone(old.values['drives'])
            self.assertFalse(old.values['direct_traps'])
            with tempfile.TemporaryDirectory() as folder:
                dest = Path(folder)/'copy.input'
                old.write(dest)
                self.assertEqual(old.values, Configuration.read(dest).values)

    def test_new_roundtrip_resolves_named_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder)/'copy.input'
            self.config.write(dest)
            actual = Configuration.read(dest)
            self.assertEqual(actual.values['groups'], self.config.values['groups'])
            np.testing.assert_array_equal(actual.reference_positions, self.config.reference_positions)
            for a, b in zip(actual.values['drives'], self.config.values['drives']):
                self.assertEqual(a['targets'], self.config.targets(b['targets']))
                self.assertEqual(a['amplitude'], b['amplitude'])

    def test_search_coordinates_and_conflicts(self):
        space = ParameterSpace(self.config, {'reference_y[2]': {'bounds':[-26,-22]},
            'drives[1].amplitude': {'bounds':[1,4]}, 'kx_per_bead[0]': {'bounds':[8,12]}})
        candidate = space.configuration([-25,3,11])
        self.assertEqual(candidate.values['reference_y'][2], -25)
        self.assertEqual(candidate.values['drives'][1]['amplitude'], 3)
        self.assertEqual(candidate.values['kx_per_bead'][0], 11)
        self.assertEqual(self.config.values['kx_per_bead'][0], 10)
        for spec in ({'F0':{'bounds':[1,4]}}, {'positions':{}},
                     {'default_lambda':{'bounds':[0,1]}},
                     {'reference_x':{'bounds':[-40,40]},'reference_x[0]':{'bounds':[-40,40]}}):
            with self.assertRaises(ValueError):
                ParameterSpace(self.config,spec)

    def test_invalid_geometry_and_drives(self):
        for override in ({'reference_x':[0]*6,'reference_y':[0]*6},
                         {'groups':{'bad':[6]}}, {'seed':1.5},
                         {'drives[0].stop':0}, {'drives[0].period':.00001}):
            with self.assertRaises(ValueError):
                self.config.updated(**override)

    def test_pair_metric_known_phase(self):
        # Two synthetic oscillators have a fixed lag but equal frequency.
        t=np.linspace(0,10,10001)
        positions=np.zeros((len(t),2,2))
        positions[:,0,0]=np.sin(2*np.pi*t)
        positions[:,1,0]=np.sin(2*np.pi*t-.6)
        analyser=MultiBeadCoherenceAnalyzer(t,positions,np.zeros((2,2)),hysteresis=.01)
        result,_=pair_metrics(analyser,1,0,2,8,1,np.pi/4)
        self.assertAlmostEqual(result['R'],1,places=6)
        self.assertAlmostEqual(result['mean_phase'],-.6,places=5)
        self.assertAlmostEqual(result['frequency_mismatch'],0,places=6)
        with self.assertRaises(ValueError):
            pair_metrics(analyser,1,0,0,1,1,np.pi/4)

    def test_protocols(self):
        for name in ('control','and','memory'):
            validate_experiment(json.loads((ROOT/f'templates/computing/{name}.json').read_text()))
        with self.assertRaises(ValueError):
            validate_experiment({'kind':'made_up'})

    @unittest.skipUnless((ROOT/'src/rowers').exists(), 'Build src/rowers for integration tests')
    def test_seeded_forces_geometry_and_empty_drives(self):
        base=self.config.updated(total_time=.02,sampling_write=1,sampling_feedback=1,drives=[])
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            a=run_simulation(base,ROOT/'src/rowers',folder/'off')
            drives=[dict(targets=[0],amplitude=2,period=1,start=0,stop=.01,phase=0,waveform='constant'),
                    dict(targets=[0,2],amplitude=3,period=1,start=0,stop=.01,phase=0,waveform='constant')]
            run_simulation(base.updated(drives=drives),ROOT/'src/rowers',folder/'on')
            off=np.loadtxt(next((folder/'off/run').glob('*.forces')))
            on=np.loadtxt(next((folder/'on/run').glob('*.forces')))
            difference=on[0,2:]-off[0,2:]
            expected=np.zeros(12);expected[0]=5;expected[5]=3
            np.testing.assert_allclose(difference,expected,atol=2e-4)
            # Legacy F0=0 and an empty independent list give identical dynamics.
            b=run_simulation(base.updated(drives=None,F0=0),ROOT/'src/rowers',folder/'legacy')
            np.testing.assert_array_equal(a.positions,b.positions)
            axes=np.loadtxt(next((folder/'off/run').glob('*.axes')))
            np.testing.assert_allclose(axes[:,1:],base.reference_positions)
            # Both thermal and deterministic trajectories are replayable.
            thermal=base.updated(temperature=1,mobility='blake',total_time=.001)
            c=run_simulation(thermal,ROOT/'src/rowers',folder/'noise1')
            d=run_simulation(thermal,ROOT/'src/rowers',folder/'noise2')
            np.testing.assert_array_equal(c.positions,d.positions)


if __name__ == '__main__':
    unittest.main()

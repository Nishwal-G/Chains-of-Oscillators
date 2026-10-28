"""Fast score regressions and synthetic search -> full validation compatibility."""
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
from transistor_fast import fast_score,validate_fast,OBJECTIVE
from simulation import save_json,load_configuration
from run_transistor_suite import main as suite_main
from optimisation import main as optimise_main
from test_transistor_device import protocol,source,fake_simulate


def spec():return json.loads((ROOT/'templates/computing/transistor_fast.json').read_text())

def metric(r):return {'status':'ok','phase_consistency_R':r,'relative_frequency_mismatch':0.,'slip_candidates_total':0}

def data(values):
    rows=[{'input':metric(.99),'outputs':{'5':metric(v),'6':metric(v)}} for v in values]
    controls=[{'outputs':{'5':metric(.05),'6':metric(.05)}} for _ in range(2)]
    return rows,controls

class FastTests(unittest.TestCase):
    def test_shape_rewards_central_transition_not_linear_or_edge_jump(self):
        c=protocol()['criteria'];s=spec()
        linear=np.array(s['fractions'])*.85+.1
        rows,controls=data(linear)
        baseline,details=fast_score(rows,controls,c,s)
        self.assertAlmostEqual(details['outputs']['5']['components']['shape'],0.,places=12)
        score,details=fast_score(*data([.1,.1,.5,.95,.95]),c,s)
        self.assertGreater(score,baseline)
        for values in ([.1,.95,.95,.95,.95],[.1,.1,.1,.1,.95],[.95]*5,[.1]*5,[.1,.7,.1,.95,.95]):
            self.assertGreater(score,fast_score(*data(values),c,s)[0])

    def test_input_leakage_slips_bad_output_and_missing_data(self):
        c=protocol()['criteria'];s=spec();rows,controls=data([.1,.1,.5,.95,.95])
        good=fast_score(rows,controls,c,s)[0]
        bad=deepcopy(rows);bad[-1]['outputs']['6']['slip_candidates_total']=2
        self.assertLess(fast_score(bad,controls,c,s)[0],good)
        bad=deepcopy(rows);bad[2]['input']['relative_frequency_mismatch']=.1
        self.assertLess(fast_score(bad,controls,c,s)[0],good)
        leak=deepcopy(controls);leak[1]['outputs']['6']=metric(.95)
        self.assertLess(fast_score(rows,leak,c,s)[0],good)
        bad=deepcopy(rows);bad[0]['outputs']['6']['status']='invalid'
        self.assertTrue(np.isfinite(fast_score(bad,controls,c,s)[0]))
        self.assertLess(fast_score(bad,controls,c,s)[0],good)
        with self.assertRaises(ValueError):validate_fast({**s,'high_index':2})
        with self.assertRaises(ValueError):validate_fast({**s,'fractions':[0,.5,.5,.8,1]})

    def test_fast_search_uses_seven_runs_and_validates_saved_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';exe=root/'fake';exe.write_text('synthetic fingerprint only')
            save_json(root/'source.json',{'configuration':source().values})
            save_json(root/'device.json',protocol())
            def command(stage,*extras):
                argv=['suite','--stage',stage,'--work',str(work),'--exe',str(exe),'--runs','1',*map(str,extras)]
                with patch.object(sys,'argv',argv):suite_main()
            command('prepare','--config',root/'source.json','--protocol',root/'device.json')
            calls=[]
            def fake(config,exe,folder,keep=False):
                calls.append(config)
                return fake_simulate(config,exe,folder,keep)
            def launch(command,check):
                self.assertTrue(check)
                with patch.object(sys,'argv',command[1:]):optimise_main()
            with patch('transistor_fast.simulate',side_effect=fake),patch('run_transistor_suite.subprocess.run',side_effect=launch):
                command('optimise-fast','--iterations','1','--runs','2','--cores','2')
            self.assertEqual(len(calls),28) # 7 simulations x 2 repeats x 2 candidates
            self.assertTrue(all(len(c.values['drives'])==2 for c in calls))
            self.assertTrue(all(c.values['total_time']==15 for c in calls))
            self.assertTrue(all(c.values['kx_per_bead']==source().values['kx_per_bead'] for c in calls))
            best=work/'search_fast/best.json';record=json.loads(best.read_text())
            self.assertEqual(record['protocol']['objective'],OBJECTIVE)
            self.assertEqual(record['protocol']['simulations_per_repeat'],7)
            self.assertEqual(record['protocol']['device']['training_disorder'],0.)
            self.assertFalse(record['protocol']['device']['objective_cycles'])
            self.assertEqual(load_configuration(best).n,7)
            self.assertFalse((work/'search').exists())
            # Full validation still runs interventions and cycles for a fast best.
            with patch('transistor_device.simulate',fake_simulate):
                command('validate','--config',best,'--out',work/'validate_fast','--points','3','--dt-divisors','1')
            folder=work/'validate_fast/dt_1/full'
            self.assertTrue((folder/'reports/phase_response.csv').exists())
            self.assertTrue((folder/'reports/switching_times.csv').exists())
            with self.assertRaises(ValueError):command('phase','--config',best,'--seed','145250000','--out',root/'badseeds')
            with self.assertRaises(ValueError):
                with patch.object(sys,'argv',['optimisation','--out',str(root/'bad'),'--config',str(work/'start.json'),
                      '--search',str(work/'search.json'),'--device',str(work/'device.json'),'--fast-device',str(work/'device.json')]):optimise_main()

if __name__=='__main__':unittest.main()

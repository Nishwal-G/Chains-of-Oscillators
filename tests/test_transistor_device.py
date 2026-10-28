"""Known-signal measurements, schedule semantics, and synthetic end-to-end CLI.

Synthetic simulator tests exercise real phase extraction and reports, not C++
physics. They must never be reported as physical validation of a transistor.
"""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from simulation import load_configuration,save_json
from switch_load import append_active_rower
from transistor_measurements import fourier_response,phase_response,probability,detect_switch,locked
from transistor_device import (validate_device,check_source,stationary_config,cycle_config,
    actual_control,device_repeat,score_device,perturb_stiffness)
from run_transistor_suite import main as suite_main
from optimisation import main as optimise_main


def protocol():
    p=json.loads((ROOT/'templates/computing/transistor_device.json').read_text())
    p['fractions']=[0,.5,1.];p['stationary']={'duration':15.,'window':[9.,13.]}
    p['phase'].update(time=6.,pre=[1.,5.],post=[9.,13.])
    p['cycles'].update(count=1,block_duration=10.,window=4.,stride=1.,dwell_windows=2)
    p['training_disorder']=.05
    p['map']={'frequency_ratios':[1.,1.1],'control_fractions':[0.,1.]}
    p['robustness']={'stiffness_fractions':[0.,.05],'temperature_offsets':[0.,.01]}
    return p


def source():
    b=load_configuration(ROOT/'templates/computing/junction.json')
    b=append_active_rower(b,10.)
    d=deepcopy(b.values['drives'])
    for v in d:v.update(period=1.,start=0.,stop=15.)
    return b.updated(drives=d,total_time=15.,seed=123)


def fake_simulate(config,exe,folder,keep=False):
    """Known oscillatory signals: no hydrodynamics, noise or simulation claim."""
    Path(folder).mkdir(parents=True,exist_ok=True)
    t=np.linspace(0,config.values['total_time'],int(config.values['total_time']*200)+1)
    dt=t[1]-t[0];d=config.values['drives'][0];omega=2*np.pi/d['period']
    inp=config.targets('input')[0];con=config.targets('control')[0]
    phase=np.zeros(len(t));input_force=np.zeros(len(t));control_force=np.zeros(len(t))
    for drive in config.values['drives']:
        active=(t>=drive['start'])&(t<drive['stop'])
        target=config.targets(drive['targets'])
        if inp in target:phase[active]=drive['phase'];input_force[active]+=drive['amplitude']
        if con in target:control_force[active]+=drive['amplitude']
    # Treat final sample using the left limit; physical measurement windows are interior.
    phase[-1]=phase[-2];input_force[-1]=input_force[-2];control_force[-1]=control_force[-2]
    theta=np.deg2rad(config.values['angles_deg']);xy=config.reference_positions
    ref=np.column_stack((xy[:,0]*np.cos(theta)+xy[:,1]*np.sin(theta),-xy[:,0]*np.sin(theta)+xy[:,1]*np.cos(theta)))
    positions=np.broadcast_to(ref,(len(t),config.n,2)).copy()
    for bead in range(config.n):
        if bead==inp:
            phi=omega*t+phase if input_force.max()>0 else 1.37*omega*t
        else:
            on=(control_force>=.49*d['amplitude'])&(input_force>0)
            freq=np.where(on,1.,1.37)*omega
            phi=np.cumsum(freq)*dt+np.where(on,phase,0.)
        positions[:,bead,0]+=np.sin(phi)
    return SimpleNamespace(time_series=t,analysis_positions=positions)


class MeasurementTests(unittest.TestCase):
    def test_fourier_and_phase_interventions(self):
        t=np.linspace(0,20,20001);q=3*np.cos(2*np.pi*t+.4)+7
        m=fourier_response(t,q,1,[3.3,13.3])
        self.assertAlmostEqual(m['amplitude'],3,places=5)
        self.assertAlmostEqual(m['phase_rad'],.4,places=5)
        phi=2*np.pi*t;shift=.6;pert=phi+np.where(t>=8,shift,0)
        follow=phase_response(t,phi,t,pert,1,[3,6],[12,17],shift)
        ignore=phase_response(t,phi,t,phi,1,[3,6],[12,17],shift)
        self.assertAlmostEqual(follow['following_quality'],1,places=8)
        self.assertAlmostEqual(follow['phase_change_rad'],shift,places=8)
        self.assertAlmostEqual(ignore['following_quality'],0,places=8)
        drift=1.17*phi
        unaffected=phase_response(t,drift,t,drift,1,[3,6],[12,17],shift)
        self.assertAlmostEqual(unaffected['following_quality'],0,places=8)
        self.assertAlmostEqual(follow['pre_difference'],0)
        pert[(t>12)&(t<13)]=np.nan
        self.assertNotEqual(phase_response(t,phi,t,pert,1,[3,6],[12,17],shift)['status'],'ok')

    def test_probability_invalids(self):
        p=probability([True,False,None,True])
        self.assertEqual((p['valid'],p['invalid'],p['fraction_all']),(3,1,.5))
        self.assertAlmostEqual(p['fraction_valid'],2/3)
        self.assertIsNone(probability([None])['fraction_valid'])

    def test_schedules_and_stiffness(self):
        base=source();p=protocol();check_source(base,p)
        conf=stationary_config(base,p,actual_control(base,p),shift=.6)
        self.assertEqual(len(conf.values['drives']),3)
        self.assertEqual(conf.values['drives'][0]['stop'],conf.values['drives'][2]['start'])
        self.assertEqual(conf.values['drives'][1]['phase'],base.values['drives'][1]['phase'])
        self.assertAlmostEqual(conf.values['drives'][2]['phase'],.6)
        scheduled,blocks=cycle_config(base,p)
        self.assertEqual([b['on'] for b in blocks],[False,True,False])
        self.assertEqual(scheduled.values['drives'][0]['amplitude'],base.values['drives'][0]['amplitude'])
        self.assertTrue(all(b['amplitude']<=.5*base.values['drives'][0]['amplitude'] for b in blocks))
        for a,b in zip(scheduled.values['drives'][1:-1],scheduled.values['drives'][2:]):self.assertEqual(a['stop'],b['start'])
        stress=perturb_stiffness(base,.1,42)
        np.testing.assert_array_equal(stress.reference_positions,base.reference_positions)
        np.testing.assert_allclose(stress.values['kx_per_bead'],perturb_stiffness(base,.1,42).values['kx_per_bead'])
        self.assertLessEqual(np.max(np.abs(np.array(stress.values['kx_per_bead'])/base.values['kx_per_bead']-1)),.1)
        with self.assertRaises(ValueError):validate_device({**p,'control_ratio':1.})

    def test_censored_switches(self):
        c=protocol()['criteria']
        def m(r):return {'status':'ok','phase_consistency_R':r,'relative_frequency_mismatch':0,'slip_candidates_total':0}
        rows=[{'start':10+i,'stop':14+i,'input':m(1),'outputs':{'5':m(.99)}} for i in range(3)]
        good=detect_switch(rows,10,20,5,True,c,2,m(.1))
        self.assertEqual(good['status'],'detected');self.assertEqual(good['delay'],4)
        self.assertEqual(detect_switch(rows,10,20,5,True,c,2,m(1))['status'],'already_in_target')
        self.assertIsNone(detect_switch([],10,20,5,True,c,2,m(.1))['delay'])

    def test_objective_penalises_wrong_output_and_phase(self):
        with tempfile.TemporaryDirectory() as tmp,patch('transistor_device.simulate',fake_simulate):
            p=protocol();record=device_repeat(source(),p,Path('/synthetic'),Path(tmp),with_cycles=True)
            score=record['score']
            broken=deepcopy(record)
            for row in broken['transfer']:row['driven']['outputs']['6']['phase_consistency_R']=.05
            self.assertLess(score_device(broken,p)[0],score)
            broken=deepcopy(record)
            for row in broken['phase']['high']:row['response']['6']['following_quality']=0.
            self.assertLess(score_device(broken,p)[0],score)
            broken=deepcopy(record);broken['transfer'][0]['driven']['outputs']['5']['status']='invalid'
            self.assertTrue(np.isfinite(score_device(broken,p)[0]))
            self.assertLess(score_device(broken,p)[0],score)


class SuiteTests(unittest.TestCase):
    def test_all_cli_stages_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp,patch('transistor_device.simulate',fake_simulate):
            root=Path(tmp);exe=root/'synthetic_executable';exe.write_text('synthetic only')
            save_json(root/'source.json',{'configuration':source().values});save_json(root/'protocol.json',protocol())
            work=root/'device'
            def command(stage,*extra):
                argv=['suite','--stage',stage,'--work',str(work),'--exe',str(exe),'--runs','1','--cores','1',*map(str,extra)]
                with patch.object(sys,'argv',argv):suite_main()
            command('prepare','--config',root/'source.json','--protocol',root/'protocol.json')
            command('baseline')
            # Exercise the suite subprocess command and the real optimiser in
            # this process, retaining the synthetic simulator patch.
            def run_search(command,check):
                self.assertTrue(check)
                with patch.object(sys,'argv',command[1:]):optimise_main()
            with patch('run_transistor_suite.subprocess.run',side_effect=run_search):
                command('optimise','--iterations','1')
            saved=json.loads((work/'search/best.json').read_text())
            self.assertEqual(saved['protocol']['objective'],'weak_control_phase_device_v1')
            self.assertIn('disorder',saved['replicates'][0])
            command('validate','--points','3','--dt-divisors','1','2')
            command('gating','--points','3')
            command('phase')
            command('cycles')
            command('locking-map')
            command('robustness')
            for name in ('baseline','validate','gating','phase','cycles','locking-map','robustness'):
                self.assertTrue((work/name/'settings.json').is_file())
            self.assertTrue((work/'validate/dt_2/full/reports/phase_response.csv').is_file())
            self.assertTrue((work/'locking-map/dt_1/locking_map_6.png').is_file())
            self.assertTrue((work/'robustness/dt_1/robustness_summary.json').is_file())
            # Search-seed leakage and output replacement must be explicit errors.
            with self.assertRaises(ValueError):command('phase','--seed','145250000','--out',root/'badseed')
            with self.assertRaises(FileExistsError):command('baseline')

if __name__=='__main__':unittest.main()

"""Test physical preservation and rejection of misleading coherence-only success."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from sweep_switch_drive import driven_config, summarise, select_source
from diagnose_switch import prepare_config
from simulation import load_configuration

class SweepTests(unittest.TestCase):
    def test_only_requested_drive_values_change(self):
        base = load_configuration(ROOT/'templates/computing/junction_diagnostic.json')
        swept = driven_config(base,2,.16,'input-only')
        for key in base.values:
            if key != 'drives':
                self.assertEqual(base.values[key],swept.values[key])
        self.assertEqual(base.values['drives'][1],swept.values['drives'][1])
        self.assertEqual(base.values['drives'][0]['amplitude'],.5)
        shared = driven_config(base,2,.16,'shared')
        self.assertEqual(shared.values['drives'][1]['period'],.16)
        for factor in (1,2,4):
            config=prepare_config(swept,'both',factor,650000)
            self.assertEqual(config.values['dt']*config.values['sampling_feedback'],
                             base.values['dt']*base.values['sampling_feedback'])
        neither=prepare_config(swept,'neither',4,650000)
        self.assertEqual([d['amplitude'] for d in neither.values['drives']],[0,0])
        self.assertEqual(neither.values['drives'][0]['period'],.16)

    def test_screening_requires_frequency_slips_and_control_checks(self):
        locked={'R':.98,'frequency_mismatch':0.,'slip_count':0}
        off={'R':.1,'frequency_mismatch':1.,'slip_count':20}
        cases={c:{'input_drive':deepcopy(locked),'output_drive':deepcopy(locked if c=='both' else off),
                  'output_input':deepcopy(locked)} for c in ('neither','input','control','both')}
        def summary(c):
            return summarise([{'status':'ok','cases':c}],.16,.9,.3,.01,0)
        self.assertEqual(summary(cases)['switch_pass_fraction'],1)
        cases['control']['output_drive']=deepcopy(locked)
        self.assertEqual(summary(cases)['switch_pass_fraction'],0)
        cases['control']['output_drive']=deepcopy(off)
        cases['both']['input_drive']['slip_count']=1
        self.assertEqual(summary(cases)['input_pass_fraction'],0)
        cases['both']['input_drive']['slip_count']=0
        cases['input']['input_drive']['frequency_mismatch']=.2
        self.assertEqual(summary(cases)['input_pass_fraction'],0)
        invalid=summarise([{'status':'invalid_phase','cases':{}}],.16,.9,.3,.01,0)
        self.assertEqual(invalid['input_pass_fraction'],0)
        self.assertIsNone(invalid['both_output_drive_R_mean'])

    def test_select_best_validation_after_folder_move(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            candidate=folder/'base_1_left_validation'/'result.json'
            candidate.parent.mkdir()
            candidate.write_text('{}')
            summary=folder/'validation_summary.json'
            summary.write_text(json.dumps([
                {'label':'base_2_left','score':-.3,'result_file':'/old/base2/result.json'},
                {'label':'base_1_left','score':-.1,'result_file':'/old/base1/result.json'}]))
            self.assertEqual(select_source(summary=summary)[0],candidate.resolve())
            candidate.unlink()
            with self.assertRaises(FileNotFoundError):
                select_source(summary=summary)

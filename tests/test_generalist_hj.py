"""Generalist CNN-PID loading and its HJ action gate."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch import nn

from training.train_online_hj import GeneralistPID, Observation, behavior_action, parse_args
from training.visual_models import RGBEncoder


class GeneralistTests(unittest.TestCase):
    def test_checkpoint_predictions_pid_reset_and_threshold(self):
        torch.set_num_threads(2)
        encoder = RGBEncoder(pretrained=False, freeze_batch_norm=True)
        encoder.network.fc = nn.Linear(512, 1)
        with torch.no_grad():
            encoder.network.fc.weight.zero_()
            encoder.network.fc.bias.fill_(8.)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'best.pth'
            torch.save({'model': encoder.network.state_dict(), 'epoch': 3}, path)
            policy = GeneralistPID(path, 'cpu', target_gap=6., kp=.5, ki=0., kd=.5)
            self.assertEqual(policy.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertFalse(policy.encoder.training)
            self.assertTrue(all(not p.requires_grad for p in policy.encoder.parameters()))
            observation = Observation((torch.full((224,224,3), 128, dtype=torch.uint8),), (1,), 10., 0)
            command, gap = policy.propose(observation, .1)
            self.assertEqual((command,gap), (1.,8.))
            policy.pid.previous_error = 100.
            policy.reset()
            self.assertEqual(policy.propose(observation,.1), (1.,8.))
            args = NS(random_steps=0, gate_start_updates=1, switching_threshold=1.)
            learner = NS(updates=1, score=lambda obs,a:.99, action=lambda obs:-3.)
            decision = behavior_action(learner,observation,command,None,10,args,False)
            self.assertTrue(decision['intervened'])
            self.assertEqual(decision['action'],-3.)
            learner.score = lambda obs,a:1.
            self.assertFalse(behavior_action(learner,observation,command,None,10,args,False)['intervened'])

    def test_generalist_arguments_do_not_require_specialist_bank(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'best.pth'
            path.touch()
            argv = ['train','--outdir',str(Path(tmp)/'run'),'--pid-checkpoint',str(path),
                    '--pid-root',str(Path(tmp)/'missing'),'--target-gap','6',
                    '--pid-kp','.9','--pid-ki','.05','--pid-kd','.15']
            with patch('sys.argv',argv):
                args = parse_args()
            self.assertEqual(args.pid_checkpoint,path)
            self.assertEqual((args.pid_kp,args.pid_ki,args.pid_kd),(.9,.05,.15))
            for extra in (['--performance-policy','sac'], ['--nominal-conditions','day_cloudy'],
                          ['--pid-kp','nan'], ['--pid-ki','-1']):
                with self.subTest(extra=extra), patch('sys.argv',argv+extra):
                    with self.assertRaises(SystemExit):
                        parse_args()


if __name__ == '__main__':
    unittest.main()

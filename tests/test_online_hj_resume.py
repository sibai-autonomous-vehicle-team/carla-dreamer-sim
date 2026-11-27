"""Checkpoint continuation restores optimization while explicitly rebuilding replay."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from training.train_online_hj import OnlineHJ, VisualNetwork, parse_args
from training.online_sage import OnlineSAGE
from tests.test_online_hj import tiny_encoder


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def args(self, policy='safety'):
        return parse_args(argv=['--outdir','/tmp/unused-resume-test',
            '--performance-policy',policy,'--safety-objective','following',
            '--challenge-gap-range','1.5','3','--no-pretrained','--freeze-encoder'])

    def batch(self):
        images=torch.ones(2,4,224,224,3,dtype=torch.uint8)
        return (images,torch.zeros(2,1),torch.tensor([[-1.],[1.]]),images,torch.zeros(2,1),
                torch.tensor([2.,2.]),torch.tensor([2.,-.2]),torch.tensor([False,True]),
                torch.zeros(2),torch.tensor([False,True]))

    def test_hj_and_sage_restore_identical_next_update_including_adam_and_targets(self):
        for cls,policy in [(OnlineHJ,'safety'),(OnlineSAGE,'sac')]:
            with self.subTest(policy=policy), TemporaryDirectory() as directory, \
                 patch.object(VisualNetwork,'__init__',tiny_encoder):
                args=self.args(policy)
                original=cls(args,'cpu')
                batch=self.batch()
                original.update(batch)
                checkpoint=Path(directory)/'checkpoint.pt'
                original.save(checkpoint,20000)
                continuation_args=deepcopy(args)
                continuation_args.resume=checkpoint
                restored=cls(continuation_args,'cpu')
                self.assertEqual(restored.restore_training(checkpoint),20000)
                self.assertEqual(restored.updates,original.updates)
                self.assertTrue(any(p.requires_grad for p in restored.actor.parameters()))
                self.assertFalse(any(p.requires_grad for p in restored.targets.parameters()))
                torch.manual_seed(123); expected=original.update(batch)
                torch.manual_seed(123); actual=restored.update(batch)
                self.assertEqual(expected,actual)
                for name in cls.model_names:
                    for key,value in getattr(original,name).state_dict().items():
                        self.assertTrue(torch.equal(value,getattr(restored,name).state_dict()[key]))

    def test_cli_inherits_training_contract_and_allows_new_runtime_settings(self):
        with TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'checkpoint.pt'
            args=self.args()
            args.gamma_start=.85; args.gamma_final=.99
            args.seed=123
            torch.save(dict(format_version=1,architecture=OnlineHJ.architecture,steps=20000,
                            config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}),checkpoint)
            continued=parse_args(argv=['--resume',str(checkpoint),'--outdir',str(Path(directory)/'new'),
                                       '--steps','40000','--port','2910','--windowed','--eval-interval','0'])
            self.assertEqual(continued.performance_policy,'safety')
            self.assertEqual(continued.safety_objective,'following')
            self.assertEqual(continued.gamma_start,.85)
            self.assertEqual(continued.seed,123)
            self.assertEqual(continued.steps,40000)
            self.assertEqual(continued.port,2910)
            self.assertEqual(continued.eval_interval,0)
            with patch('sys.stderr'),self.assertRaises(SystemExit):
                parse_args(argv=['--resume',str(checkpoint),'--outdir','/tmp/new',
                                 '--steps','40000','--safety-objective','collision'])
            with patch('sys.stderr'),self.assertRaises(SystemExit):
                parse_args(argv=['--resume',str(checkpoint),'--outdir','/tmp/new','--steps','20000'])


if __name__ == '__main__': unittest.main()

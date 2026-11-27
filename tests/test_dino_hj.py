"""DINO trainable-block boundaries, real RL gradients and exact continuation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

from training.train_dino_hj import DinoHJ,parse_args
from training.train_online_hj import Observation,Replay,Transition


class TinyDino(nn.Module):
    """Cheap differentiable backbone with the same CLS/block interface."""
    embed_dim=8
    chunked_blocks=False

    def __init__(self):
        super().__init__()
        self.patch_embed=nn.Linear(3,8)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(8,8),nn.GELU()) for _ in range(4)])
        self.norm=nn.LayerNorm(8)

    def forward_features(self,x):
        x=self.patch_embed(x.mean((2,3)))
        for block in self.blocks:x=block(x)
        return dict(x_norm_clstoken=self.norm(x))


class DinoTests(unittest.TestCase):
    def args(self,*extra):
        return parse_args(['--outdir','/tmp/unused-dino-test','--cpu',*extra])

    def test_selected_blocks_update_and_frozen_weights_stay_identical(self):
        torch.set_num_threads(2)
        with patch('training.train_dino_hj.load_backbone',return_value=TinyDino()):
            learner=DinoHJ(self.args(),'cpu')
        actor=learner.actor
        before={n:p.detach().clone() for n,p in actor.named_parameters()}
        for i,block in enumerate(actor.network.blocks):
            self.assertTrue(all(p.requires_grad==(i>=2) for p in block.parameters()))
        self.assertFalse(any(p.requires_grad for p in actor.network.patch_embed.parameters()))
        self.assertFalse(any(p.requires_grad for p in learner.targets.parameters()))
        self.assertEqual([g['lr'] for g in learner.actor_optimizer.param_groups],[1e-4,1e-5])
        image=torch.randint(1,255,(224,224,3),dtype=torch.uint8)
        a=Observation((image,)*4,(1,1,1,1),15.,0)
        b=Observation((image,)*3+(image//2,),(1,1,1,2),15.,0)
        replay=Replay(10)
        replay.extend([Transition(a,1.,b,2.,1.8,False),Transition(a,-1.,b,.1,-.1,True)])
        batch=replay.sample(4,'cpu')
        metrics=learner.update(batch)
        self.assertEqual((metrics['gamma'],metrics['alpha']),(.85,.05))
        for name,param in actor.named_parameters():
            if not param.requires_grad:self.assertTrue(torch.equal(before[name],param),name)
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in actor.named_parameters() if n.startswith('network.blocks.3')))
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in actor.named_parameters() if n.startswith('head.')))
        self.assertFalse(actor.network.training)
        diagnostic=learner.action_diagnostics(a,1.)
        self.assertTrue(torch.isfinite(torch.tensor(diagnostic['deterministic_q'])))
        with TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            learner.save(path,20000)
            continued=parse_args(['--resume',str(path),'--outdir',directory+'/continued','--steps','100000','--cpu'])
            with patch('training.train_dino_hj.load_backbone',return_value=TinyDino()):
                restored=DinoHJ(continued,'cpu')
            self.assertEqual(restored.restore_training(path),20000)
            learner.environment_steps=20000
            torch.manual_seed(19);expected=learner.update(batch)
            torch.manual_seed(19);actual=restored.update(batch)
            self.assertEqual(expected,actual)
            payload=torch.load(path,weights_only=True)
            self.assertEqual(payload['preprocessing']['kind'],'dinov2_cls_rgb')
            self.assertEqual(payload['preprocessing']['unfreeze_blocks'],2)

    def test_gamma_uses_environment_steps_independently_of_entropy_updates(self):
        # Exercise the schedule without loading a visual backbone.
        learner=DinoHJ.__new__(DinoHJ)
        learner.args=self.args()
        learner.updates=10000
        for step,expected in ((0,.85),(10000,.864),(50000,.92),(100000,.99),(120000,.99)):
            learner.environment_steps=step
            self.assertAlmostEqual(learner.scheduled_gamma(),expected)
            self.assertEqual(learner.schedule(learner.args.alpha,learner.args.alpha_final),.05)
        learner.args.gamma_anneal_steps=None
        learner.args.anneal_updates=10000
        self.assertEqual(learner.scheduled_gamma(),.99)
        with self.assertRaises(SystemExit):self.args('--gamma-anneal-steps','0')

    def test_history_order_is_preserved_and_shape_validated(self):
        with patch('training.train_dino_hj.load_backbone',return_value=TinyDino()):
            learner=DinoHJ(self.args(),'cpu')
        image=torch.stack([torch.full((224,224,3),v,dtype=torch.uint8) for v in (20,60,100,200)])[None]
        speed=torch.zeros(1,1)
        with torch.no_grad():
            first=learner.actor.features(image,speed).reshape(1,4,8)
            swapped=learner.actor.features(image.flip(1),speed).reshape(1,4,8)
        torch.testing.assert_close(first.flip(1),swapped)
        with self.assertRaises(ValueError):learner.actor.features(image.float(),speed)

    def test_bad_modes_and_resume_training_changes_are_rejected(self):
        with self.assertRaises(ValueError):self.args('--freeze-encoder')
        with self.assertRaises(ValueError):self.args('--no-pretrained')
        with self.assertRaises(ValueError):self.args('--dino-unfreeze-blocks','13')


if __name__=='__main__':unittest.main()

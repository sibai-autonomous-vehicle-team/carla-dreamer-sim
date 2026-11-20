"""RGB alignment, actor architecture, and BatchNorm target-state invariants."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from training.closed_loop import run_episode

import torch

from training.config import Settings
from training.dataset import hj_examples
from training.visual_dataset import ImageReplay
from training.visual_models import ResNetActor,ResNetCritic,PREPROCESS
from training.visual_ddpg import soft_update_visual


class VisualTests(unittest.TestCase):
    def test_live_rgb_actor_receives_image_not_numeric_features(self):
        sent,seen = [],[]
        frame_image = torch.full((224,224,3),77,dtype=torch.uint8)
        world = NS(get_snapshot=lambda: NS(frame=len(sent)),
                   get_settings=lambda: NS(synchronous_mode=True,fixed_delta_seconds=.1))
        env = NS(ego=NS(get_world=lambda:world),nonego=NS(id=1),
                 _config=NS(action=NS(discrete=False)),
                 get_rgb_observation=lambda frame: frame_image)
        def step(action):
            sent.append(action)
            return None,0.,False,{}
        env.step=step
        def predict(image):
            seen.append(image)
            return (0.,.1)
        policy=NS(observation_kind='rgb',settings={'dt':.1,'safety_buffer_m':1.,'max_gap_m':9.},
                  predict_safe_action=predict)
        def snapshot(_env,_dt):
            pose=torch.zeros(12);pose[6]=10
            return len(sent),len(sent)*.1,pose,5.
        with patch('training.closed_loop.read_snapshot',snapshot):
            run_episode(env,policy,lambda e:(1.,0.),mode='actor_only',scenario_id='rgb',
                        seed=1,condition='day_sunny',max_steps=2)
        self.assertEqual(len(seen),1)
        self.assertIs(seen[0],frame_image)
        self.assertEqual(sent,[(1.,0.),(0.,.1)])

    def test_resnet_shape_preprocessing_bounds_and_action_gradient(self):
        torch.set_num_threads(2)
        actor = ResNetActor().eval()
        critic = ResNetCritic().eval().requires_grad_(False)
        image = torch.zeros(1,224,224,3,dtype=torch.uint8)
        expected = -torch.tensor(PREPROCESS['mean'])/torch.tensor(PREPROCESS['std'])
        torch.testing.assert_close(actor.preprocess(image)[0,:,0,0],expected)
        self.assertEqual(actor.network.fc.in_features,512)
        self.assertEqual(actor.network.fc.out_features,2)
        action = actor(image)
        self.assertEqual(action.shape,(1,2))
        self.assertLessEqual(float(action[0,0].abs()),3)
        self.assertLessEqual(float(action[0,1].abs()),1)
        critic(image,action).sum().backward()
        self.assertIsNotNone(actor.network.fc.weight.grad)
        self.assertGreater(float(actor.network.fc.weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):
            actor(torch.zeros(1,8))

    def test_target_updates_batchnorm_buffers_and_counters(self):
        online = torch.nn.BatchNorm2d(3)
        target = torch.nn.BatchNorm2d(3)
        online.running_mean.fill_(4.)
        online.num_batches_tracked.fill_(8)
        soft_update_visual(target,online,.25)
        torch.testing.assert_close(target.running_mean,torch.ones(3))
        self.assertEqual(target.num_batches_tracked.item(),8)

    def test_image_rows_match_vector_targets_and_combined_source_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root/'day_sunny'
            (condition/'obses').mkdir(parents=True)
            n = 5
            fields = dict(states=[torch.zeros(n,12),torch.ones(n,12)],
                          actions=[torch.zeros(n,2),torch.ones(n,2)],
                          distances=[torch.tensor([4.,5.,9.,6.,5.])]*2,
                          costs=[torch.zeros(n)]*2,seq_lengths=torch.tensor([n,n]))
            for name,value in fields.items(): torch.save(value,condition/f'{name}.pth')
            for episode in range(2):
                image = torch.arange(n,dtype=torch.uint8).view(n,1,1,1).expand(n,224,224,3).clone()+100*episode
                torch.save(image,condition/'obses'/f'episode_{episode:03d}.pth')
            (root/'merge_manifest.json').write_text(json.dumps({'sources':[
                {'folder':'old','count_per_condition':1},{'folder':'new','count_per_condition':1}]}))
            data = {k:torch.cat(fields[k]) for k in ('states','actions','distances','costs')}
            data.update(lengths=fields['seq_lengths'],sources=['old','new'],conditions=['day_sunny']*2,
                        episode_ids=torch.tensor([0,0]),source_paths={'old':'/unused/old','new':'/unused/new'})
            cfg = Settings()
            replay = ImageReplay(data,[0,1],cfg,root)
            self.assertEqual(len(replay),4)
            batch = replay.batch(torch.arange(4))
            self.assertEqual(batch[0][:,0,0,0].tolist(),[1,2,101,102])
            self.assertEqual(batch[2][:,0,0,0].tolist(),[2,3,102,103])
            numerical = hj_examples(data,[0,1],cfg)
            for i in (1,3,4,5): torch.testing.assert_close(batch[i],numerical[i])
            self.assertEqual(batch[5].tolist(),[False,True,False,True])
            torch.testing.assert_close(replay.bc_target[:,0],torch.tensor([-.5,3.,-.5,3.]))
            self.assertEqual(replay.bc_mask.tolist(),[True,False,True,False])
            from training.visual_algorithms import LearnerOptions
            from dataclasses import replace
            # Logged zero/one controls differ from the deterministic PID here;
            # strict recorded cloning must reject them rather than copying noise.
            recorded=ImageReplay(data,[0,1],cfg,root,teacher_options=replace(LearnerOptions(),bc_target='recorded',recorded_tolerance=0.))
            self.assertFalse(recorded.bc_mask.any())


if __name__ == '__main__': unittest.main()

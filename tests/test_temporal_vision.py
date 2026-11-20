"""Temporal ordering, episode padding, and live/replay observation parity."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from training.visual_models import ResNetActor, ResNetCritic
from training.visual_dataset import ImageReplay
from training.closed_loop import run_episode


class TemporalTests(unittest.TestCase):
    def test_replay_history_and_successor_never_use_future_or_other_episode(self):
        replay = object.__new__(ImageReplay)
        replay.history_frames = 5
        frames = torch.arange(8,dtype=torch.uint8).view(8,1,1,1).expand(8,224,224,3)
        self.assertEqual(replay.observation(frames,0)[:,0,0,0].tolist(),[0]*5)
        self.assertEqual(replay.observation(frames,2)[:,0,0,0].tolist(),[0,0,0,1,2])
        self.assertEqual(replay.observation(frames,5)[:,0,0,0].tolist(),[1,2,3,4,5])
        self.assertEqual(replay.observation(frames,6)[:,0,0,0].tolist(),[2,3,4,5,6])

    def test_temporal_actor_critic_shapes_and_gradients(self):
        torch.set_num_threads(2)
        actor = ResNetActor(history_frames=3,freeze_batch_norm=True).train()
        critic = ResNetCritic(history_frames=3,freeze_batch_norm=True).eval().requires_grad_(False)
        frames = torch.randint(256,(1,3,224,224,3),dtype=torch.uint8)
        action = actor(frames)
        self.assertEqual(tuple(action.shape),(1,2))
        self.assertTrue((action.abs()<=torch.tensor([3.,1.])).all())
        critic(frames,action).sum().backward()
        self.assertGreater(float(actor.temporal_head[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(actor.network.conv1.weight.grad.abs().sum()),0.)
        with self.assertRaises(ValueError): actor(frames[:,0])

    def test_live_history_matches_replay_padding_and_resets_each_episode(self):
        sent,seen=[],[]
        world=NS(get_snapshot=lambda:NS(frame=len(sent)),
                 get_settings=lambda:NS(synchronous_mode=True,fixed_delta_seconds=.1))
        env=NS(ego=NS(get_world=lambda:world),nonego=NS(id=1),
               _config=NS(action=NS(discrete=False)),
               get_rgb_observation=lambda frame:torch.full((224,224,3),frame,dtype=torch.uint8))
        def step(action):
            sent.append(action)
            return None,0.,False,{}
        env.step=step
        def predict(images):
            seen.append(images[:,0,0,0].tolist())
            return (0.,0.)
        policy=NS(observation_kind='rgb',history_frames=5,
                  settings={'dt':.1,'safety_buffer_m':1.,'max_gap_m':9.},predict_safe_action=predict)
        def snapshot(env,dt):
            pose=torch.zeros(12);pose[6]=10
            return len(sent),len(sent)*dt,pose,5.
        with patch('training.closed_loop.read_snapshot',snapshot):
            for _ in range(2):
                run_episode(env,policy,lambda e:(1.,0.),mode='actor_only',scenario_id='test',
                            seed=1,condition='day_sunny',max_steps=3)
        self.assertEqual(seen,[[0,0,0,0,1],[0,0,0,1,2],[3,3,3,3,4],[3,3,3,4,5]])

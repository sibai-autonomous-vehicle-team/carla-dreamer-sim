"""Kinematic state semantics, frame alignment, online updates and continuation."""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from training.train_state_hj import StateHistory, StateHJ, STATE_FEATURES
from training.train_online_hj import Observation, Replay, Transition, parse_args


class StateHJTests(unittest.TestCase):
    def environment(self,yaw=0.,origin=(0.,0.)):
        angle=math.radians(yaw); c,s=math.cos(angle),math.sin(angle)
        vector=lambda n:NS(x=n*c,y=n*s,z=0.)
        def actor(leader):
            location=NS(x=origin[0]+(10*c if leader else 0),y=origin[1]+(10*s if leader else 0),z=0.)
            return NS(get_transform=lambda:NS(location=location,rotation=NS(yaw=yaw)),
                      get_velocity=lambda:vector(12. if leader else 15.),
                      get_acceleration=lambda:vector(-2. if leader else 0.))
        actors={1:actor(False),2:actor(True)}
        snapshot=NS(frame=10,find=actors.get)
        route=NS(project=lambda location:(0.,0.),point=lambda progress:([*origin,0.],yaw),lengths=[0.,100.])
        return NS(world=NS(get_snapshot=lambda:snapshot),ego=NS(id=1),nonego=NS(id=2),tracker=lambda actor:route)

    def test_state_contains_motion_and_is_invariant_to_world_translation_rotation(self):
        with patch('training.closed_loop.read_snapshot',return_value=(10,1.,None,6.)):
            first=StateHistory(1,0,.1)
            obs=first.capture(self.environment())
            rotated=StateHistory(1,0,.1).capture(self.environment(90.,(100.,200.)))
            torch.testing.assert_close(obs.frames[0],torch.tensor([.6,.5,-.3,0.,0.,0.,1.,0.,-.2,0.,0.,0.]))
            torch.testing.assert_close(obs.frames[0],rotated.frames[0])
            self.assertEqual(len(obs.frames[0]),len(STATE_FEATURES))
            with self.assertRaisesRegex(RuntimeError,'skipped'):first.capture(self.environment())

    def test_actor_critics_update_and_checkpoint_can_continue(self):
        torch.set_num_threads(2)
        args=parse_args(argv=['--outdir','/tmp/unused-state-test','--performance-policy','safety',
            '--safety-objective','following','--challenge-gap-range','1.5','3','--history-frames','1',
            '--no-pretrained','--cpu'])
        learner=StateHJ(args,'cpu')
        a=Observation((torch.ones(12),),(10,),15.,0)
        b=Observation((torch.ones(12)*.9,),(11,),14.,0)
        replay=Replay(8)
        replay.extend([Transition(a,1.,b,3.,2.9,False),Transition(a,-1.,b,.1,-.1,True)])
        actor_before=learner.actor.head[-1].weight.detach().clone()
        critic_before=learner.critics[0].head[-1].weight.detach().clone()
        batch=replay.sample(8,'cpu')
        metrics=learner.update(batch)
        self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        self.assertFalse(torch.equal(actor_before,learner.actor.head[-1].weight))
        self.assertFalse(torch.equal(critic_before,learner.critics[0].head[-1].weight))
        self.assertTrue(math.isfinite(learner.action_diagnostics(a,1.)['brake_q']))
        with TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'state.pt';learner.save(checkpoint,20000)
            with patch('training.train_online_hj.OnlineHJ',StateHJ):
                continued_args=parse_args(argv=['--resume',str(checkpoint),'--outdir',directory+'/new','--steps','100000'])
            restored=StateHJ(continued_args,'cpu')
            self.assertEqual(restored.restore_training(checkpoint),20000)
            torch.manual_seed(7);expected=learner.update(batch)
            torch.manual_seed(7);actual=restored.update(batch)
            self.assertEqual(expected,actual)


if __name__=='__main__':unittest.main()

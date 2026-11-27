"""Both PID proposers use full braking at the critic gate, without actor calls."""
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

import torch

from run_gap_pid import GapPID
from training.compare_generalist_specialist import ContextSelectorPID
from training.evaluate_online_hj import rollout
from training.train_online_hj import OnlineHJ, behavior_action


class FullBrakeTests(unittest.TestCase):
    def learner(self,policy='pid'):
        learner=OnlineHJ.__new__(OnlineHJ)
        learner.args=NS(performance_policy=policy,switching_threshold=1.)
        learner.updates=1000
        learner.score=Mock(return_value=.5)
        learner.action=Mock(side_effect=AssertionError('PID fallback must not call the SAC actor'))
        return learner

    def test_training_and_inference_pid_gates_never_call_actor(self):
        learner=self.learner()
        args=NS(performance_policy='pid',random_steps=0,gate_start_updates=500,switching_threshold=1.)
        for proposal in (1.5,-1.2,-3.):
            self.assertEqual(learner.filter_action(None,proposal)['action'],-3.)
            decision=behavior_action(learner,None,proposal,None,1000,args,False)
            self.assertEqual(decision['action'],-3.)
            self.assertEqual(decision['source'],'full_brake_fallback')
            self.assertTrue(decision['intervened'])
        learner.action.assert_not_called()
        learner.score.return_value=1.
        self.assertEqual(learner.filter_action(None,1.5)['action'],1.5)
        self.assertFalse(behavior_action(learner,None,1.5,None,1000,args,False)['intervened'])

    def test_both_pid_proposers_execute_full_brake_in_rollout(self):
        world=NS(frame=100)
        world.get_snapshot=lambda:NS(frame=world.frame,
            timestamp=NS(delta_seconds=.1,elapsed_seconds=world.frame*.1),
            find=lambda actor:NS(get_velocity=lambda:NS(x=10.,y=0.,z=0.)))
        executed=[]
        class Env:
            def __init__(self,*a,**kw):
                self.world=world;self.ego=NS(id=1);self.nonego=NS(id=2)
                self.events=[];self.departures=[];self.leader_events=[]
            def tracker(self,actor):return NS(steering=lambda:0.)
            def get_rgb_observation(self,frame):return torch.ones(224,224,3,dtype=torch.uint8)
            def step(self,action):executed.append(action[0]);world.frame+=1
            def measurements(self):return dict(collision_events=[],lane_departure_frames=[])
            def close(self):pass
        class Nominal:
            profile=dict(target_gap=6.)
            def reset(self):self.pid=GapPID(6.,.5,0.,.5)
            def propose(self,observation,dt):return self.pid.update(8.,dt),8.
        selector=ContextSelectorPID(NS(score=lambda condition,gap:{'day_cloudy':1.}),
                                    NS(gap=8.),{'day_cloudy':NS(gap=8.)},'day_cloudy')
        args=NS(condition='day_cloudy',dt=.1,history_frames=4,speed_range=[10.,20.],
                brake_duration=2.,episode_steps=4,collision_grace=0.,safety_objective='collision',
                max_gap=9.,max_separation=50.,event_failure_margin=1.,switching_threshold=1.)
        case=dict(case=0,seed=12,scenario={},brake_after=1.,
                  start=dict(gap=4.,closing_speed=0.,leader_speed=10.,ego_speed=10.))
        learner=self.learner()
        with patch('training.closed_loop.read_snapshot',side_effect=lambda *a:(world.frame,world.frame*.1,None,4.)), \
             patch('training.train_online_hj.planned_lane_status',return_value=dict(inside=True)), \
             patch.object(ContextSelectorPID,'estimate',side_effect=lambda p,o:p.gap):
            for policy in (Nominal(),selector):
                for mode in ('filter','safety'):
                    executed.clear()
                    result=rollout(learner,policy,None,Env,args,case,mode)
                    self.assertEqual(executed,[-3.]*4)
                    self.assertEqual(result['intervention_fraction'],1. if mode=='filter' else 0.)
                    self.assertTrue(result['horizon_complete'])
                executed.clear()
                rollout(learner,policy,None,Env,args,case,'pid_then_safety')
                self.assertEqual(executed[:3],[-3.]*3)
                self.assertGreater(executed[3],0.)  # One deliberate nominal-action probe.
        learner.action.assert_not_called()

    def test_full_brake_request_maps_to_one_hundred_percent_brake(self):
        from car_dreamer.carla_base_env import CarlaBaseEnv
        control=CarlaBaseEnv.get_vehicle_control(NS(_config=NS(action=NS(discrete=False))),(-3.,0.))
        self.assertEqual(control.brake,1.)
        self.assertEqual(control.throttle,0.)

    def test_separate_sac_goal_policy_retains_its_actor_path(self):
        learner=self.learner('sac')
        learner.action=Mock(return_value=-1.4)
        self.assertEqual(learner.filter_action(None,1.)['action'],-1.4)
        learner.action.assert_called_once()


if __name__=='__main__':unittest.main()

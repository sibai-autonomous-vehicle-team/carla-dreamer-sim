"""Episode stop bounds are independent from HJ constraints and first-hit labels."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from training.train_online_hj import (gap_failure,gap_termination,margin,first_hit_rows,
                                    hj_backup,Observation,Transition,parse_args,SafetyPolicy)
from training.evaluate_online_hj import rollout


class GapTerminationTests(unittest.TestCase):
    def test_separate_bounds_preserve_negative_margin_and_terminal_backup(self):
        args=NS(safety_objective='following',min_gap=1.,max_gap=9.,termination_gap_range=[.2,20.])
        for gap in [.8,9.2]:
            self.assertTrue(gap_failure(gap,args))
            self.assertLess(margin(gap,args),0)
            self.assertFalse(gap_termination(gap,args))
        self.assertTrue(gap_termination(.2,args))
        self.assertTrue(gap_termination(20.,args))
        args.termination_gap_range=None
        self.assertTrue(gap_termination(.8,args))
        self.assertTrue(gap_termination(9.2,args))

    def test_replay_keeps_first_failed_transition_even_if_rollout_recovers(self):
        obs=Observation((),(10,),15.,0)
        rows=[Transition(obs,0.,obs,.2,-.2,True),Transition(obs,0.,obs,-.2,.1,False)]
        kept=first_hit_rows(rows)
        self.assertEqual(len(kept),1)
        target=hj_backup(torch.tensor([kept[0].margin]),torch.tensor([kept[0].next_margin]),
                         torch.tensor([True]),torch.tensor([99.]),.99)
        self.assertAlmostEqual(float(target),-.196,places=6)

    def test_cli_accepts_independent_stops(self):
        args=parse_args(argv=['--outdir','/tmp/unused-gap-test','--performance-policy','safety',
            '--safety-objective','following','--challenge-gap-range','1.5','3',
            '--termination-gap-range','.2','20'])
        self.assertEqual(args.termination_gap_range,[.2,20.])
        self.assertEqual((args.min_gap,args.max_gap),(1.,9.))

    def test_evaluation_continues_after_violation_and_recovery_still_counts_as_failure(self):
        torch.set_num_threads(2)
        world=NS(frame=100)
        world.get_snapshot=lambda:NS(frame=world.frame,
            timestamp=NS(delta_seconds=.1,elapsed_seconds=world.frame*.1),
            find=lambda actor:NS(get_velocity=lambda:NS(x=15.,y=0.,z=0.)))
        class Env:
            def __init__(self,*args,**kwargs):
                self.world=world;self.ego=NS(id=1);self.nonego=NS(id=2)
                self.events=[];self.departures=[];self.leader_events=[]
            def tracker(self,actor): return NS(steering=lambda:0.)
            def get_rgb_observation(self,frame):return torch.ones(224,224,3,dtype=torch.uint8)
            def step(self,action):world.frame+=1
            def measurements(self):return dict(collision_events=[],lane_departure_frames=[])
            def close(self):pass
        gaps=[5.,9.2,8.,7.,6.]
        learner=NS(action=lambda observation,deterministic=False:0.,score=lambda observation,action:0.)
        args=NS(condition='day_cloudy',brake_duration=2.,speed_range=[10.,20.],history_frames=4,
            dt=.1,episode_steps=4,collision_grace=0.,safety_objective='following',min_gap=1.,max_gap=9.,
            termination_gap_range=[.2,20.],max_separation=50.,event_failure_margin=1.,switching_threshold=1.)
        case=dict(case=0,seed=1,start=dict(gap=5.,closing_speed=0.,leader_speed=15.,ego_speed=15.),
                  brake_after=1.,scenario={})
        with patch('training.closed_loop.read_snapshot',side_effect=lambda *a:(world.frame,0.,None,gaps[world.frame-100])), \
             patch('training.train_online_hj.planned_lane_status',return_value=dict(inside=True)):
            result=rollout(learner,SafetyPolicy(learner,6.),None,Env,args,case,'safety')
        self.assertEqual(result['ticks'],4)
        self.assertTrue(result['failure'])
        self.assertFalse(result['horizon_complete'])
        self.assertTrue(result['rollout_horizon_complete'])
        self.assertAlmostEqual(result['minimum_margin'],-.2)


if __name__=='__main__':unittest.main()

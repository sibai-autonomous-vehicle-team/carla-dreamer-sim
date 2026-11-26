"""Online HJ target semantics, camera contracts, replay isolation and gradients."""
from types import SimpleNamespace as NS
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import math
import random
import json
import io
from contextlib import redirect_stdout, ExitStack
from unittest.mock import patch
import unittest

import torch
from torch import nn

from training.train_online_hj import (FrameHistory, Observation, Transition, Replay,
                                     OnlineHJ, VisualNetwork, hj_backup, finalize_episode,
                                     TrainingViewer, with_preview, margin, gap_failure,
                                     route_speed_limit, with_route_aware_leader, planned_lane_status, RandomExplorer,
                                     behavior_action,sample_start,with_closing_starts)


def tiny_encoder(self, history, include_speed, pretrained, freeze_encoder):
    nn.Module.__init__(self)
    self.freeze_batch_norm = True
    self.history_frames, self.include_speed = history, include_speed
    self.width = 512*history + int(include_speed)
    self.network = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3,512))
    self.register_buffer('image_mean',torch.zeros(1,3,1,1))
    self.register_buffer('image_std',torch.ones(1,3,1,1))
    if freeze_encoder: self.network.requires_grad_(False)


class OnlineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_random_exploration_holds_commands_and_covers_braking_and_acceleration(self):
        explorer=RandomExplorer((5,5),random.Random(42))
        commands=[explorer.action() for _ in range(100)]
        for start in range(0,100,5):
            self.assertEqual(len(set(commands[start:start+5])),1)
        self.assertTrue(any(a>2 for a in commands))
        self.assertTrue(any(a<-2 for a in commands))
        self.assertTrue(all(-3<=a<=3 for a in commands))

    def test_behavior_switches_pid_to_safety_and_preserves_deliberate_exploration(self):
        args=NS(random_steps=1000,gate_start_updates=500,switching_threshold=.25)
        learner=NS(updates=0,score=lambda obs,a:.1,action=lambda obs:-3.)
        explorer=NS(action=lambda:2.5)
        self.assertEqual(behavior_action(learner,None,1.,explorer,0,args,False)['source'],'warmup_random')
        self.assertEqual(behavior_action(learner,None,1.,explorer,1100,args,False)['action'],1.)
        learner.updates=500
        decision=behavior_action(learner,None,1.,explorer,1100,args,False)
        self.assertEqual(decision['source'],'safety_actor'); self.assertTrue(decision['intervened'])
        self.assertEqual(decision['action'],-3.)
        learner.score=lambda obs,a:.5
        self.assertEqual(behavior_action(learner,None,1.,explorer,1100,args,False)['source'],'contextual_pid')
        decision=behavior_action(learner,None,1.,explorer,1100,args,True)
        self.assertEqual(decision['source'],'coverage_random'); self.assertFalse(decision['intervened'])

    def test_start_sampler_covers_close_closing_and_normal_starts_with_bounded_speeds(self):
        args=NS(challenge_fraction=.5,closing_speed_range=[1.,6.],speed_range=[10.,20.],
                challenge_gap_range=[.5,3.],gap_range=[2.,8.],exploration_episode_fraction=.25)
        rng=random.Random(42)
        starts=[sample_start(args,rng) for _ in range(100)]
        self.assertTrue(any(s['challenge'] for s in starts)); self.assertTrue(any(not s['challenge'] for s in starts))
        self.assertTrue(any(s['exploration_episode'] for s in starts))
        for s in starts:
            self.assertTrue(10<=s['leader_speed']<=s['ego_speed']<=20)
            self.assertAlmostEqual(s['ego_speed']-s['leader_speed'],s['closing_speed'])
            if s['challenge']:
                self.assertTrue(.5<=s['gap']<=3.)
                self.assertTrue(1<=s['closing_speed']<=6.)
            else:self.assertEqual(s['closing_speed'],0.)

    def test_closing_initializer_sets_different_actual_vehicle_speeds(self):
        env=with_closing_starts(object)()
        env.args=NS(ego_initial_speed=18.)
        env.dt=.1; env.carla=NS(VehicleControl=lambda **kw:NS(**kw),Vector3D=lambda **kw:NS(**kw))
        def actor():
            a=NS(speed=0.,get_location=lambda:None,apply_control=lambda control:None,
                 get_transform=lambda:NS(get_forward_vector=lambda:NS(x=1.,y=0.,z=0.)))
            a.set_target_velocity=lambda v:setattr(a,'speed',math.sqrt(v.x**2+v.y**2+v.z**2))
            a.get_velocity=lambda:NS(length=lambda:a.speed)
            return a
        env.ego,env.nonego=actor(),actor()
        env.validate_lanes=lambda:None
        env.world=NS(tick=lambda:None)
        env.tracker=lambda actor:self.route()
        env.initialize_speed(12.)
        self.assertEqual(env.ego.speed,18.); self.assertEqual(env.nonego.speed,12.)

    def test_training_loop_executes_pid_and_records_behavior_and_measured_starts(self):
        from training.train_online_hj import main
        world=NS(frame=100,get_settings=lambda:NS(),apply_settings=lambda value:None)
        world.get_snapshot=lambda:NS(frame=world.frame,timestamp=NS(delta_seconds=.1,elapsed_seconds=world.frame*.1),
            find=lambda actor:NS(get_velocity=lambda:NS(x=15.,y=0.,z=0.)))
        actions=[]
        class Env:
            def __init__(self,*args,**kwargs):
                self.world=world; self.events=[]; self.departures=[]; self.leader_events=[]
                self.ego=NS(id=1); self.nonego=NS(id=2)
            def tracker(self,actor):return NS(steering=lambda:0.)
            def get_rgb_observation(self,frame):return torch.ones(224,224,3,dtype=torch.uint8)
            def step(self,action):actions.append(action); world.frame+=1
            def measurements(self):return dict(collision_events=[],lane_departure_frames=[])
            def close(self):pass
        learner=NS(updates=0,score=lambda obs,action:1.5,save=lambda path,steps:path.touch())
        def update(batch):
            learner.updates+=1
            return dict(updates=learner.updates,critic_mse=0.,gamma=.85,alpha=.05)
        learner.update=update
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root=Path(directory); bank=root/'bank'/'day_cloudy'; bank.mkdir(parents=True)
            for name in ('pid.json','cnn.pth'):(bank/name).touch()
            pid=NS(folder=bank,sha256='test',profile=dict(target_gap=6.,kp=.9,ki=.05,kd=.15),
                   reset=lambda:None,propose=lambda observation,dt:(1.2,7.))
            scenario=dict(full_route=[[float(i),0.,0.,0.,0.,0.] for i in range(30)])
            argv=['train','--outdir',str(root/'run'),'--pid-root',str(root/'bank'),'--steps','2',
                  '--episode-steps','1','--random-steps','0','--gate-start-updates','0',
                  '--challenge-fraction','0','--exploration-episode-fraction','0',
                  '--learning-starts','1','--batch-size','1','--capacity','8','--cpu']
            argv.extend(['--eval-interval','1'])
            stack.enter_context(patch('sys.argv',argv))
            stack.enter_context(patch.dict('sys.modules',{
                'carla':NS(Client=lambda *args:NS(set_timeout=lambda value:None,load_world=lambda name:world)),
                'training.collect_braking':NS(start_server=lambda args:None),
                'collect_parallel':NS(stop=lambda process:None)}))
            for target,value in [('training.train_online_hj.OnlineHJ',lambda *args:learner),
                                 ('training.train_online_hj.ContextPID',lambda *args:pid),
                                 ('training.train_online_hj.with_route_aware_leader',lambda *args:Env),
                                 ('training.train_online_hj.with_closing_starts',lambda base,*args:base),
                                 ('training.train_online_hj.planned_lane_status',lambda *args:dict(inside=True)),
                                 ('training.evaluate_closed_loop.make_scenarios',lambda *args:[scenario]),
                                 ('training.closed_loop.read_snapshot',lambda *args:(world.frame,world.frame*.1,None,3.))]:
                stack.enter_context(patch(target,side_effect=value))
            with patch('training.evaluate_online_hj.evaluate') as evaluation:
                with redirect_stdout(io.StringIO()):main()
                self.assertEqual([call.args[6] for call in evaluation.call_args_list],[0,1,2])
            self.assertFalse(list((root/'run').glob('checkpoint_[0-9]*.pt')))
            self.assertTrue((root/'run'/'checkpoint_final.pt').exists())
            records=[json.loads(line) for line in (root/'run'/'transitions.jsonl').read_text().splitlines()]
            self.assertEqual(actions,[(1.2,0.),(1.2,0.)])
            self.assertTrue(all(r['behavior']['source']=='contextual_pid' for r in records))
            self.assertEqual(learner.updates,1)
            self.assertEqual(len((root/'run'/'measured_starts.jsonl').read_text().splitlines()),2)

    def env(self):
        env = NS(frame=100, image=torch.ones(224,224,3,dtype=torch.uint8), ego=NS(id=1))
        def snapshot():
            return NS(frame=env.frame, timestamp=NS(elapsed_seconds=env.frame*.1,delta_seconds=.1),
                      find=lambda actor: NS(get_velocity=lambda: NS(x=3.,y=4.,z=0.)))
        env.world=NS(get_snapshot=snapshot)
        def rgb(frame):
            if frame != env.frame: raise RuntimeError('stale camera')
            return env.image
        env.get_rgb_observation=rgb
        return env

    def test_histories_pad_then_shift_and_keep_capture_storage_immutable(self):
        env=self.env(); history=FrameHistory(4,7)
        first=history.capture(env)
        env.image.fill_(2); env.frame+=1
        second=history.capture(env)
        self.assertEqual(first.frame_ids,(100,100,100,100))
        self.assertEqual(second.frame_ids,(100,100,100,101))
        self.assertEqual(first.speed,5.)
        self.assertEqual(int(first.frames[-1][0,0,0]),1)
        self.assertIs(first.frames[-1],second.frames[-2])

    def test_identical_pixels_are_allowed_when_capture_ids_are_fresh(self):
        env=self.env(); h=FrameHistory(4,0)
        h.capture(env); env.frame+=1
        self.assertEqual(h.capture(env).frame_ids[-1],101)

    def test_preview_attaches_before_first_tick_without_advancing_extra_frames(self):
        calls=[]
        class Base:
            def __init__(self): self.frame=100; self.tick()
            def tick(self):
                self.frame+=1; calls.append(('tick',self.frame)); return self.frame
        viewer=NS(attach=lambda env:calls.append(('attach',env.frame)),
                  synchronize=lambda frame:calls.append(('sync',frame)))
        env=with_preview(Base,viewer)(); env.tick()
        self.assertEqual(calls,[('attach',100),('tick',101),('sync',101),('tick',102),('sync',102)])

    def test_preview_preserves_native_rgb_pixels_and_rejects_mismatched_frame(self):
        import numpy as np
        with patch.dict(os.environ,{'SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy'}):
            viewer=TrainingViewer()
            try:
                row=self.transition().observation
                viewer.chase_frame=row.frame_ids[-1]
                viewer.chase=np.zeros((720,1280,3),dtype=np.uint8)
                params=dict(step=1,action=.2,steering=0.,gap=3.,q=.1,updates=0,phase='Test',metrics=None)
                viewer.show(row,**params)
                self.assertEqual(tuple(viewer.window.get_at((1328,55)))[:3],(1,1,1))
                viewer.show(row,actor_active=True,**params)
                pixels=viewer.pygame.surfarray.array3d(viewer.window)[1292:,399:425]
                self.assertTrue(((pixels[:,:,0]>200)&(pixels[:,:,1]<120)&(pixels[:,:,2]<120)).any())
                viewer.chase_frame+=1
                with self.assertRaisesRegex(RuntimeError,'do not match'): viewer.show(row,**params)
            finally: viewer.close()

    def route(self, curved=False):
        points = ([[20*math.sin(i*.05),20*(1-math.cos(i*.05)),0.] for i in range(100)]
                  if curved else [[float(i),0.,0.] for i in range(100)])
        lengths=[0.]
        for a,b in zip(points,points[1:]): lengths.append(lengths[-1]+math.hypot(b[0]-a[0],b[1]-a[1]))
        return NS(route=points,lengths=lengths,index=0,project=lambda location:(0.,0.),steering=lambda:.1)

    def test_leader_slows_for_curves_and_retains_straight_cruise_speed(self):
        self.assertEqual(route_speed_limit(self.route(),None,15.),15.)
        speed=route_speed_limit(self.route(True),None,15.)
        self.assertAlmostEqual(speed,math.sqrt(2.*20.),places=4)

    def test_leader_brakes_when_above_curve_speed_and_preserves_emergency_pulse(self):
        cls=with_route_aware_leader(object)
        env=cls()
        env.args=NS(initial_speed=15.,leader_brake_after=1.,leader_brake_duration=2.)
        env.initial_frame=100; env.dt=.1
        env.world=NS(get_snapshot=lambda:NS(frame=100))
        env.nonego=NS(get_location=lambda:None,get_velocity=lambda:NS(length=lambda:15.))
        env.tracker=lambda actor:self.route(True)
        env.carla=NS(VehicleControl=lambda **kw:NS(**kw))
        self.assertGreater(env.leader_control().brake,0.)
        env.world.get_snapshot=lambda:NS(frame=115)
        self.assertEqual(env.leader_control().brake,1.)

    def test_leader_rejects_infeasible_fast_start_on_sharp_curve(self):
        from training.braking_scenario import InvalidBrakingScenario
        env=with_route_aware_leader(object)()
        env.nonego=NS(get_location=lambda:None)
        env.tracker=lambda actor:self.route(True)
        with self.assertRaisesRegex(InvalidBrakingScenario,'exceeds curve preview limit'):
            env.initialize_speed(15.)

    def test_planned_lane_check_catches_leader_on_another_driving_lane(self):
        actor=NS(get_transform=lambda:NS(location=None,rotation=NS(yaw=0.)),
                 bounding_box=NS(extent=NS(x=2.,y=1.)))
        env=NS(tracker=lambda actor:NS(project=lambda location:(0.,2.),point=lambda progress:(None,0.)),
               map=NS(get_waypoint=lambda *args,**kwargs:NS(lane_width=3.5)),carla=NS(LaneType=NS(Driving=1)))
        self.assertFalse(planned_lane_status(env,actor)['inside'])

    def test_skipped_and_stale_captures_fail(self):
        env=self.env(); h=FrameHistory(4,0); h.capture(env); env.frame+=2
        with self.assertRaisesRegex(RuntimeError,'Nonconsecutive'): h.capture(env)
        env=self.env()
        env.get_rgb_observation=lambda frame: (_ for _ in ()).throw(RuntimeError('stale camera'))
        with self.assertRaisesRegex(RuntimeError,'stale camera'): FrameHistory(4,0).capture(env)

    def transition(self):
        env=self.env(); h=FrameHistory(4,0); a=h.capture(env); env.frame+=1; b=h.capture(env)
        return Transition(a,.5,b,2.,1.,False)

    def test_replay_rejects_reset_crossing_and_wrong_history(self):
        row=self.transition(); replay=Replay(3)
        row.successor=Observation(row.successor.frames,row.successor.frame_ids,5.,1)
        with self.assertRaisesRegex(ValueError,'reset'): replay.extend([row])
        row=self.transition()
        row.successor=Observation(row.successor.frames,(90,100,100,101),5.,0)
        with self.assertRaisesRegex(ValueError,'shift'): replay.extend([row])

    def test_failure_absorbs_but_time_limit_bootstraps(self):
        result=hj_backup(torch.tensor([2.,2.]),torch.tensor([-.2,1.]),
                         torch.tensor([True,False]),torch.tensor([9.,.5]),.9)
        torch.testing.assert_close(result,torch.tensor([.02,.65]))

    def test_collision_objective_does_not_fail_at_positive_gap_or_large_gap(self):
        args=NS(safety_objective='collision',min_gap=1.,max_gap=9.)
        self.assertEqual(margin(.5,args),.5)
        self.assertEqual(margin(20.,args),9.)
        self.assertFalse(gap_failure(.5,args))
        self.assertFalse(gap_failure(0.,args))
        self.assertFalse(gap_failure(50.,args))
        row=self.transition()
        row.next_margin=margin(.5,args); row.failed=row.next_margin<=0
        rows=finalize_episode([row],[],[],1.)
        self.assertFalse(rows[-1].failed)
        rows=finalize_episode(rows,[(101,20)],[],1.)
        self.assertTrue(rows[-1].failed)
        self.assertEqual(rows[-1].next_margin,-1.)

    def test_collision_only_lane_truncation_does_not_create_negative_safety_label(self):
        row=self.transition()
        rows=finalize_episode([row],[],[101],1.,include_lane_failure=False)
        self.assertFalse(rows[-1].failed)
        self.assertEqual(rows[-1].next_margin,1.)
        rows=finalize_episode(rows,[(101,20)],[101],1.,include_lane_failure=False)
        self.assertTrue(rows[-1].failed)
        self.assertEqual(rows[-1].next_margin,-1.)

    def test_following_objective_preserves_explicit_old_failure_boundaries(self):
        args=NS(safety_objective='following',min_gap=1.,max_gap=9.)
        self.assertEqual(margin(.5,args),-.5)
        self.assertEqual(margin(10.,args),-1.)
        self.assertTrue(gap_failure(.5,args))

    def test_collision_following_preserves_contact_experience_and_penalizes_losing_leader(self):
        args=NS(safety_objective='collision-following',min_gap=1.,max_gap=9.)
        self.assertEqual(margin(.2,args),.2)
        self.assertFalse(gap_failure(.2,args))
        self.assertFalse(gap_failure(0.,args))
        self.assertEqual(margin(10.,args),-1.)
        self.assertTrue(gap_failure(10.,args))
        self.assertLess(margin(8.,args),margin(5.,args))

    def test_positive_but_below_epsilon_intervenes_without_changing_hj_labels(self):
        learner=OnlineHJ.__new__(OnlineHJ)
        learner.args=NS(switching_threshold=1.)
        learner.action=lambda observation,deterministic: -3.
        learner.score=lambda observation,action: .5
        result=learner.filter_action(None,2.)
        self.assertTrue(result['intervened'])
        self.assertEqual(result['action'],-3.)
        learner.score=lambda observation,action: 1.
        self.assertEqual(learner.filter_action(None,2.)['action'],2.)
        # Positive successor margins are not absorbing failures merely because
        # a controller-intervention threshold is larger than the margin.
        target=hj_backup(torch.tensor([.2]),torch.tensor([.1]),torch.tensor([False]),torch.tensor([.05]),.9)
        torch.testing.assert_close(target,torch.tensor([.065]))

    def test_late_collision_trims_episode_before_replay_admission(self):
        env=self.env(); h=FrameHistory(4,0); observation=h.capture(env); rows=[]
        for _ in range(3):
            env.frame+=1; successor=h.capture(env)
            rows.append(Transition(observation,0.,successor,2.,2.,False)); observation=successor
        rows=finalize_episode(rows,[(102,20)],[],1.)
        self.assertEqual(len(rows),2)
        self.assertTrue(rows[-1].failed)
        self.assertEqual(rows[-1].next_margin,-1.)

    def test_online_update_changes_actor_and_critics_and_keeps_targets_frozen(self):
        args=NS(history_frames=4,include_speed=True,no_pretrained=True,freeze_encoder=True,
                actor_lr=1e-3,critic_lr=1e-3,gamma_start=.85,gamma_final=.9999,
                alpha=.05,alpha_final=0.,anneal_updates=100,tau=.05)
        with patch.object(VisualNetwork,'__init__',tiny_encoder): learner=OnlineHJ(args,'cpu')
        replay=Replay(3); replay.extend([self.transition()])
        before=learner.actor.head[-1].weight.detach().clone()
        critic_before=learner.critics[0].head[-1].weight.detach().clone()
        metrics=learner.update(replay.sample(2,'cpu'))
        self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())
        self.assertFalse(torch.equal(before,learner.actor.head[-1].weight))
        self.assertFalse(torch.equal(critic_before,learner.critics[0].head[-1].weight))
        self.assertFalse(any(p.requires_grad for p in learner.targets.parameters()))
        self.assertFalse(any(p.requires_grad for c in learner.critics for p in c.network.parameters()))
        self.assertTrue(all(p.grad is None for c in learner.critics for p in c.network.parameters()))
        self.assertTrue(-3 <= learner.action(self.transition().observation,True) <= 3)

    def test_checkpoint_loads_frozen_and_camera_only_ignores_speed(self):
        args=NS(history_frames=4,include_speed=False,no_pretrained=True,freeze_encoder=False,
                actor_lr=1e-3,critic_lr=1e-3,gamma_start=.85,gamma_final=.9999,
                alpha=.05,alpha_final=0.,anneal_updates=100,tau=.05)
        with patch.object(VisualNetwork,'__init__',tiny_encoder), TemporaryDirectory() as directory:
            learner=OnlineHJ(args,'cpu')
            observation=self.transition().observation
            changed=Observation(observation.frames,observation.frame_ids,99.,observation.episode)
            self.assertEqual(learner.action(observation,True),learner.action(changed,True))
            score=learner.score(observation,1.)
            path=Path(directory)/'checkpoint.pt'; learner.save(path,10)
            frozen=OnlineHJ.from_checkpoint(path)
            self.assertEqual(score,frozen.score(observation,1.))
            self.assertFalse(any(p.requires_grad for p in frozen.actor.parameters()))


if __name__ == '__main__': unittest.main()

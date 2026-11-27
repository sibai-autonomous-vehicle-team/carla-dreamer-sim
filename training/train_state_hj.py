"""Single online HJ actor-critic with synchronized, ego-relative kinematic state.

Uses the visual trainer's transitions, first-hit target, action mapping and
evaluation. No PINN, PID, vision encoder, or separate performance actor is used.
Steering remains the fixed route tracker. This is a privileged-state diagnostic.
"""
import math
from unittest.mock import patch

import torch
from torch import nn

from . import train_online_hj as hj


STATE_FEATURES = (
    ('gap_m',10.), ('ego_forward_speed_mps',30.),
    ('leader_minus_ego_forward_speed_mps',10.), ('relative_lateral_position_m',3.),
    ('relative_lateral_velocity_mps',5.), ('sin_relative_yaw',1.),
    ('cos_relative_yaw',1.), ('ego_forward_acceleration_mps2',10.),
    ('leader_forward_acceleration_mps2',10.), ('signed_lane_offset_m',2.),
    ('lane_heading_error_rad',math.pi), ('route_curvature_ahead_per_m',.1),
)


class StateActor(nn.Module):
    forward = hj.Actor.forward
    sample_features = hj.Actor.sample_features

    def __init__(self, *contract):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(len(STATE_FEATURES),128),nn.ReLU(),
            nn.Linear(128,128),nn.ReLU(),nn.Linear(128,2))
        nn.init.uniform_(self.head[-1].weight,-.003,.003)
        nn.init.zeros_(self.head[-1].bias)

    def features(self, states, speed): return states[:,-1]


class StateCritic(nn.Module):
    value = hj.Critic.value
    forward = hj.Critic.forward

    def __init__(self,*contract,value_bounds=None):
        super().__init__()
        self.value_bounds=value_bounds
        self.state_head=nn.Sequential(nn.Linear(len(STATE_FEATURES),128),nn.LayerNorm(128),nn.ReLU())
        self.action_head=nn.Sequential(nn.Linear(1,32),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(160,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))

    def features(self, states, speed): return states[:,-1]


class StateHJ(hj.OnlineHJ):
    architecture='online_hj_state_kinematic12_v1'

    def __init__(self,args,device):
        with patch.multiple(hj,Actor=StateActor,Critic=StateCritic):
            super().__init__(args,device)

    def save(self,path,steps):
        contract=dict(kind='kinematic_state',history=1,
                      features=[dict(name=name,scale=scale) for name,scale in STATE_FEATURES])
        with patch.object(hj,'PREPROCESS',contract):
            super().save(path,steps)


def wrap_angle(value): return math.atan2(math.sin(value),math.cos(value))


class StateHistory:
    def __init__(self,length,episode,dt):
        if length != 1: raise ValueError('State trainer requires --history-frames 1')
        self.episode,self.dt,self.last_frame=episode,dt,None

    def capture(self,env):
        from .closed_loop import read_snapshot
        frame,_,_,gap=read_snapshot(env,self.dt)
        snapshot=env.world.get_snapshot()
        if snapshot.frame != frame or (self.last_frame is not None and frame != self.last_frame+1):
            raise RuntimeError('State observation skipped a simulator frame')
        ego,leader=(snapshot.find(actor.id) for actor in (env.ego,env.nonego))
        et,lt=ego.get_transform(),leader.get_transform()
        ev,lv=ego.get_velocity(),leader.get_velocity()
        ea,la=ego.get_acceleration(),leader.get_acceleration()
        yaw=math.radians(et.rotation.yaw)
        c,s=math.cos(yaw),math.sin(yaw)
        forward=lambda v: v.x*c+v.y*s
        lateral=lambda x,y: -x*s+y*c
        relative_yaw=math.radians(lt.rotation.yaw-et.rotation.yaw)
        route=env.tracker(env.ego)
        progress,_=route.project(et.location)
        point,route_yaw=route.point(progress)
        route_angle=math.radians(route_yaw)
        offset=-(et.location.x-point[0])*math.sin(route_angle)+(et.location.y-point[1])*math.cos(route_angle)
        ahead=min(progress+10.,route.lengths[-1]-1e-3)
        if ahead<=progress: raise RuntimeError('Insufficient remaining route for state observation')
        _,ahead_yaw=route.point(ahead)
        curvature=wrap_angle(math.radians(ahead_yaw-route_yaw))/(ahead-progress)
        values=(gap,forward(ev),forward(lv)-forward(ev),
            lateral(lt.location.x-et.location.x,lt.location.y-et.location.y),
            lateral(lv.x-ev.x,lv.y-ev.y),math.sin(relative_yaw),math.cos(relative_yaw),
            forward(ea),forward(la),offset,wrap_angle(yaw-route_angle),curvature)
        state=torch.tensor([v/scale for v,(_,scale) in zip(values,STATE_FEATURES)],dtype=torch.float32)
        if not torch.isfinite(state).all(): raise ValueError('Nonfinite kinematic state')
        if env.world.get_snapshot().frame != frame: raise RuntimeError('World advanced during state capture')
        self.last_frame=frame
        speed=math.sqrt(ev.x**2+ev.y**2+ev.z**2)
        return hj.Observation((state,),(frame,),speed,self.episode)


def main():
    from .braking_scenario import BrakingScenario

    class StateScenario(BrakingScenario):
        def __init__(self,client,config,*rest,**kwargs):
            config.camera=False
            super().__init__(client,config,*rest,**kwargs)

    with patch.multiple(hj,OnlineHJ=StateHJ,FrameHistory=StateHistory), \
         patch('training.braking_scenario.BrakingScenario',StateScenario):
        args=hj.parse_args(dict(performance_policy='safety',history_frames=1,
            no_pretrained=True,cpu=True,steps=100000,random_steps=1000,learning_starts=1000,
            safety_objective='following',challenge_gap_range=[1.5,3.],
            gamma_start=.85,gamma_final=.99,alpha=.05,alpha_final=0.,anneal_updates=10000,
            port=2930),description=__doc__)
        if args.performance_policy!='safety' or args.history_frames!=1 or args.windowed:
            raise ValueError('Use safety-only mode, --history-frames 1, and no --windowed for state training')
        hj.main(args,manifest_overrides=dict(
            actor_inputs='12 normalized kinematic/route features from the action-time snapshot',
            critic_inputs='same state + longitudinal request',
            state_features=[dict(name=name,scale=scale) for name,scale in STATE_FEATURES],
            sensor_contract='consecutive synchronized snapshots; camera disabled',
            preprocessing='kinematic features divided by declared scales',
            critic_fusion='state Linear128/LayerNorm/ReLU + action Linear32/ReLU',
            comparison='privileged-state control; match objective, scenarios, updates and evaluation seeds to RGB'))


if __name__=='__main__': main()

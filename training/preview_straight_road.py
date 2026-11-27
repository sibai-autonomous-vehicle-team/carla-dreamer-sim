"""Render actual, matched 224x224 CARLA observations for all nine conditions.

Uses a dedicated server. The vehicles are stationary and frozen for a matched
visibility comparison; training still samples moving and braking trajectories.
Raw images use the same RGB sensor, pose, resolution and FOV as online training.
"""
import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from .straight_road import create_world,make_scenarios


CONDITIONS=[f'{time}_{weather}' for time in ('day','sunset','night')
            for weather in ('sunny','cloudy','rainy')]


def contact_sheet(folder):
    sheet=Image.new('RGB',(3*244,3*260),(20,24,30))
    draw=ImageDraw.Draw(sheet)
    for index,condition in enumerate(CONDITIONS):
        x,y=(index%3)*244,(index//3)*260
        draw.text((x+10,y+8),condition.replace('_',' '),fill=(245,245,245))
        with Image.open(folder/f'{condition}.png') as raw:
            sheet.paste(raw,(x+10,y+28))
    sheet.save(folder/'all_conditions.png')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,default=Path('training/_local')/
                   datetime.now().strftime('straight_road_preview_%Y%m%d_%H%M%S'))
    p.add_argument('--port',type=int,default=2950)
    p.add_argument('--start-server',action='store_true')
    p.add_argument('--server',type=Path,default=Path('.local/carla-0.9.15/CarlaUE4.sh'))
    p.add_argument('--gap',type=float,default=5.)
    p.add_argument('--blueprint',default='vehicle.audi.tt',help='Use a vehicle with working headlights and tail lights')
    p.add_argument('--settle-frames',type=int,default=60)
    p.add_argument('--road-length',type=float,default=1000.)
    args=p.parse_args()
    if args.gap<=0 or args.settle_frames<4: p.error('Positive gap and at least four settling frames required')
    args.outdir.mkdir(parents=True,exist_ok=False)
    import carla
    from .straight_weather import make_weather,with_straight_weather,CAMERA_ATTRIBUTES
    from .braking_scenario import BrakingScenario
    from .collect_braking import start_server
    from collect_parallel import stop
    process=env=world=original_settings=None
    args.road_environment='straight'
    try:
        if args.start_server: process=start_server(args)
        client=carla.Client('127.0.0.1',args.port);client.set_timeout(120.)
        current=client.get_world()
        if list(current.get_actors().filter('vehicle.*')):
            raise RuntimeError('Preview requires its own idle CARLA server; found existing vehicles')
        world=create_world(client,args.road_length,args.outdir/'road.xodr')
        original_settings=world.get_settings()
        scenario=make_scenarios(world,1,42,100)[0]
        config=SimpleNamespace(condition='day_sunny',blueprint=args.blueprint,
            initial_gap=args.gap,initial_speed=None,ego_initial_speed=0.,
            stationary_leader_test=False,leader_brake_test=False,
            leader_brake_after=2.,leader_brake_duration=2.,graphics='vulkan',camera=True,real_time=False,
            handoff_speed_bounds=[0.,.1],handoff_gap_bounds=[args.gap-.1,args.gap+.1])
        env=with_straight_weather(BrakingScenario)(client,config,.1,scenario,reload_world=False)
        for actor in (env.ego,env.nonego):actor.set_simulate_physics(False)
        reference={name:dict(x=actor.get_location().x,y=actor.get_location().y,z=actor.get_location().z,
                            yaw=actor.get_transform().rotation.yaw)
                   for name,actor in [('ego',env.ego),('leader',env.nonego)]}
        records=[]
        for condition in CONDITIONS:
            env.args.condition=condition
            world.set_weather(make_weather(condition))
            env.apply_lights()
            history=deque(maxlen=4)
            for _ in range(args.settle_frames):
                frame=env.tick()
                pixels=env.get_rgb_observation(frame).numpy().copy()
                history.append((frame,pixels))
            frame,pixels=history[-1]
            if pixels.shape!=(224,224,3) or pixels.dtype!=np.uint8:
                raise RuntimeError('Capture differs from model RGB contract')
            Image.fromarray(pixels).save(args.outdir/f'{condition}.png')
            for index,(capture,rgb) in enumerate(history):
                Image.fromarray(rgb).save(args.outdir/f'{condition}_history_{index}.png')
            pose={name:dict(x=actor.get_location().x,y=actor.get_location().y,z=actor.get_location().z,
                           yaw=actor.get_transform().rotation.yaw)
                  for name,actor in [('ego',env.ego),('leader',env.nonego)]}
            if pose!=reference:raise RuntimeError('Vehicle poses changed between configurations')
            record=dict(condition=condition,frame=frame,history_frames=[f for f,_ in history],
                rgb_shape=list(pixels.shape),pixel_mean=float(pixels.mean()),
                weather=str(world.get_weather()),ego_lights=int(env.ego.get_light_state()),
                measured_gap_m=env.get_distance(),vehicle_poses=pose)
            records.append(record);print(json.dumps(record),flush=True)
        manifest=dict(camera=dict(width=224,height=224,fov=120,location=dict(x=1.5,z=2.),
                                  exposure_attributes=CAMERA_ATTRIBUTES,
                                  postprocessing='CARLA sensor rendering with shared auto-exposure range; raw RGB PNG, no resizing or enhancement'),
            road='straight asphalt, center line, grass verges; no buildings or other traffic',
            vehicle=args.blueprint,
            preview='stationary frozen vehicles; matched poses across conditions, not training data',
            settling_frames=args.settle_frames,records=records)
        (args.outdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        contact_sheet(args.outdir)
        print(f'CAPTURES: {args.outdir.resolve()}',flush=True)
    finally:
        if env is not None:env.close()
        if world is not None and original_settings is not None:world.apply_settings(original_settings)
        if process is not None:stop(process)


if __name__=='__main__':main()

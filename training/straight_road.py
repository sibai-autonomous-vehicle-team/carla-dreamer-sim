"""Minimal CARLA OpenDRIVE road for controlled visual following experiments."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def opendrive(length=1000.):
    if not math.isfinite(length) or length<400:
        raise ValueError('Straight road length must be finite and at least 400 m')
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="ControlledStraightFollowing" version="1.00" date="2026-11-26" north="20" south="-20" east="{length}" west="0">
    <geoReference><![CDATA[+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs]]></geoReference>
  </header>
  <road name="StraightFollowing" length="{length}" id="1" junction="-1">
    <link/>
    <type s="0" type="rural"><speed max="30" unit="m/s"/></type>
    <planView><geometry s="0" x="0" y="0" hdg="0" length="{length}"><line/></geometry></planView>
    <elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile>
    <lateralProfile/>
    <lanes><laneOffset s="0" a="0" b="0" c="0" d="0"/>
      <laneSection s="0">
        <left><lane id="1" type="shoulder" level="false"><link/><width sOffset="0" a="6" b="0" c="0" d="0"/></lane></left>
        <center><lane id="0" type="none" level="false"><link/><roadMark sOffset="0" type="solid" weight="standard" color="white" width="0.15" laneChange="none"/></lane></center>
        <right>
          <lane id="-1" type="driving" level="false"><link/><width sOffset="0" a="3.6" b="0" c="0" d="0"/><roadMark sOffset="0" type="solid" weight="standard" color="white" width="0.15" laneChange="none"/></lane>
          <lane id="-2" type="shoulder" level="false"><link/><width sOffset="0" a="6" b="0" c="0" d="0"/></lane>
        </right>
      </laneSection>
    </lanes>
    <objects/><signals/>
  </road>
</OpenDRIVE>
'''


def create_world(client,length=1000.,save_path=None):
    import carla
    xml=opendrive(length)
    if save_path is not None: Path(save_path).write_text(xml)
    world=client.generate_opendrive_world(xml,carla.OpendriveGenerationParameters(
        vertex_distance=2.,max_road_length=50.,wall_height=0.,additional_width=.6,
        smooth_junctions=False,enable_mesh_visibility=False))
    original=world.get_settings()
    settings=world.get_settings()
    settings.synchronous_mode=True
    settings.fixed_delta_seconds=.1
    world.apply_settings(settings)
    try:
        add_road_surface(world,length)
    finally:
        world.apply_settings(original)
    return world


def color_texture(pixels):
    import carla
    height,width=pixels.shape[:2]
    texture=carla.TextureColor(width,height)
    for y in range(height):
        for x in range(width):
            texture.set(x,y,carla.Color(*map(int,pixels[y,x]),255))
    return texture


def matte_surface(world,names):
    """Replace the reused prop material's normal and packed surface maps."""
    import carla
    for parameter,channels in (
        (carla.MaterialParameter.Normal,(.5,.5,1.,1.)),
        (carla.MaterialParameter.AO_Roughness_Metallic_Emissive,(1.,1.,0.,0.)),
    ):
        texture=carla.TextureFloatColor(4,4)
        for y in range(4):
            for x in range(4):texture.set(x,y,carla.FloatColor(*channels))
        world.apply_float_color_texture_to_objects(names,parameter,texture)


def add_road_surface(world,length):
    """Packaged flat meshes with runtime textures; OpenDRIVE retains physics."""
    import carla
    import numpy as np
    bp=world.get_blueprint_library().find('static.prop.mesh')
    bp.set_attribute('mesh_path','/Game/Carla/Static/Other/00_Opt/SM_Plane_Opt.SM_Plane_Opt')
    # The mesh material tiles its UVs in metres, so use separate geometry for
    # the asphalt and grass rather than painting a road mask into a tiled image.
    world.tick()
    before=set(world.get_names_of_all_objects())
    bp.set_attribute('scale','.6')  # horizontal face: 6 m along x, 7.2 m across y
    for x in range(3,math.ceil(length)+3,6):
        actor=world.spawn_actor(bp,carla.Transform(carla.Location(x=x,z=-.065),carla.Rotation(pitch=-90)))
        actor.set_collisions(False)
    world.tick()
    names=list(set(world.get_names_of_all_objects())-before)
    rng=np.random.default_rng(42)
    asphalt=np.repeat(np.clip(rng.normal(43,3,(128,128,1)),0,255).astype(np.uint8),3,axis=2)
    world.apply_color_texture_to_objects(names,carla.MaterialParameter.Diffuse,color_texture(asphalt))
    matte_surface(world,names)
    bp.set_attribute('scale','1000')
    before=set(world.get_names_of_all_objects())
    ground=world.spawn_actor(bp,carla.Transform(carla.Location(x=length/2,z=-100.25),carla.Rotation(pitch=-90)))
    ground.set_collisions(False)
    world.tick()  # Object-name inventory is updated on the server frame barrier.
    grass=list(set(world.get_names_of_all_objects())-before)
    world.apply_color_texture_to_objects(grass,carla.MaterialParameter.Diffuse,
        color_texture(np.full((4,4,3),[49,90,29],dtype=np.uint8)))
    matte_surface(world,grass)
    bp.set_attribute('mesh_path','/Game/Carla/Static/RoadLine/St_Road_TileRoad_LaneMarkingBroken.St_Road_TileRoad_LaneMarkingBroken')
    bp.set_attribute('scale','1')
    sample=world.spawn_actor(bp,carla.Transform(carla.Location(x=-100,z=-100)))
    box=sample.bounding_box
    long=max(box.extent.x,box.extent.y)
    short=min(box.extent.x,box.extent.y)
    sample.destroy()
    if long<=0 or short/long>.15:
        raise RuntimeError('Packaged center-line mesh has unexpected dimensions')
    scale=3/(2*long)
    bp.set_attribute('scale',str(scale))
    yaw=90. if box.extent.y>box.extent.x else 0.
    before=set(world.get_names_of_all_objects())
    for x in range(3,math.ceil(length),9):
        actor=world.spawn_actor(bp,carla.Transform(carla.Location(x=x,z=.005+box.extent.z*scale),carla.Rotation(yaw=yaw)))
        actor.set_collisions(False)
    world.tick()
    markings=list(set(world.get_names_of_all_objects())-before)
    world.apply_color_texture_to_objects(markings,carla.MaterialParameter.Diffuse,
        color_texture(np.full((4,4,3),[225,218,175],dtype=np.uint8)))
    print(f'Road materials: {len(names)} asphalt tiles, {len(grass)} grass plane, {len(markings)} marking tiles',flush=True)


def make_scenarios(world,count,seed,steps):
    """Identical straight geometry for every seed; dynamics are randomized upstream."""
    from .evaluate_closed_loop import pose_values
    road_map=world.get_map()
    road=ET.fromstring(road_map.to_opendrive()).find("road[@id='1']")
    if road is None:
        raise ValueError('Controlled straight road 1 was not loaded')
    length=float(road.attrib['length'])
    route=[]
    for distance in range(50,int(length)-50,2):
        waypoint=road_map.get_waypoint_xodr(1,-1,float(distance))
        if waypoint is None: break
        route.append(pose_values(waypoint.transform))
    # The training handoff uses route[20]. Leave room for the full episode.
    if len(route)<max(140,int(steps*1.5)):
        raise ValueError('Straight road is too short for this episode; increase --straight-road-length')
    result=[]
    for index in range(count):
        ego,leader=list(route[0]),list(route[5])
        ego[2]+=.5;leader[2]+=.5
        result.append(dict(id=f'straight_{index:03d}',seed=seed+index,ego=ego,leader=leader,
                           route=[list(p) for p in route[5:]],full_route=[list(p) for p in route]))
    return result


def scenarios_for_args(world,count,seed,steps,args):
    if getattr(args,'road_environment','town')=='straight':
        return make_scenarios(world,count,seed,steps)
    from .evaluate_closed_loop import make_scenarios as town_scenarios
    return town_scenarios(world,count,seed,steps)

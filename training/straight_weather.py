"""Explicit visual conditions and vehicle lighting for the controlled road only."""

CAMERA_ATTRIBUTES=dict(exposure_mode='histogram',exposure_min_bright='4.0',
    exposure_max_bright='16.0',exposure_speed_up='5.0',exposure_speed_down='5.0',
    exposure_compensation='-0.7',bloom_intensity='0.0',lens_flare_intensity='0.0',
    motion_blur_intensity='0.0',blur_amount='0.0',fstop='8.0',focal_distance='1000.0')


def make_weather(condition):
    import carla
    time,kind=condition.split('_')
    if time not in ('day','sunset','night') or kind not in ('sunny','cloudy','rainy'):
        raise ValueError(f'Unknown condition: {condition}')
    weather=carla.WeatherParameters()
    weather.sun_altitude_angle={'day':60.,'sunset':8.,'night':-30.}[time]
    # Side lighting avoids staring into the noon sun along the +X road.
    weather.sun_azimuth_angle=270. if time=='day' else 200.
    weather.cloudiness={'sunny':0.,'cloudy':60.,'rainy':80.}[kind]
    weather.precipitation=90. if kind=='rainy' else 0.
    weather.precipitation_deposits=80. if kind=='rainy' else 0.
    weather.wetness=100. if kind=='rainy' else 0.
    weather.wind_intensity=50. if kind=='rainy' else 5.
    weather.fog_density=0. if kind=='sunny' else 2.
    weather.fog_distance=100. if kind=='sunny' else .75
    weather.fog_falloff=.1
    weather.scattering_intensity=1.
    weather.mie_scattering_scale=.03
    weather.rayleigh_scattering_scale=.0331
    return weather


def light_state(condition,braking=False):
    import carla
    light=carla.VehicleLightState
    state=light.NONE
    if not condition.startswith('day_') or condition.endswith('_rainy'):
        state=light.Position|light.LowBeam
    if braking:state|=light.Brake
    return carla.VehicleLightState(state)


def vehicle_lights(actor,condition,braking=False):
    actor.set_light_state(light_state(condition,braking))


def with_straight_weather(base):
    """Same weather, lights and particle-following setup in train, eval and previews."""
    class StraightWeatherScenario(base):
        def __init__(self,client,args,*positional,**kwargs):
            from copy import copy
            args=copy(args)
            args.camera_attributes=CAMERA_ATTRIBUTES.copy()
            args.weather_parameters=make_weather(args.condition)
            args.vehicle_initial_lights=light_state(args.condition,True)
            super().__init__(client,args,*positional,**kwargs)
            self.world.set_weather(make_weather(args.condition))
            self.apply_lights()

        def apply_lights(self):
            for actor in (self.ego,self.nonego):
                vehicle_lights(actor,self.args.condition,actor.get_control().brake>.01)

        def tick(self):
            # CARLA's precipitation effects follow the spectator. Keep it by the
            # active ego even for offscreen rendering, without extra world ticks.
            if self.ego is not None:self.follow_ego()
            return super().tick()

        def leader_control(self):
            control=super().leader_control()
            vehicle_lights(self.nonego,self.args.condition,control.brake>.01)
            return control

        def step(self,action):
            vehicle_lights(self.ego,self.args.condition,action[0]<0)
            return super().step(action)
    return StraightWeatherScenario

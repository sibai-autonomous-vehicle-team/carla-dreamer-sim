from types import SimpleNamespace as NS
import unittest

import carla

from training.straight_weather import make_weather,vehicle_lights,with_straight_weather,CAMERA_ATTRIBUTES


class StraightWeatherTests(unittest.TestCase):
    def test_weather_conditions_differ_without_mutating_shared_presets(self):
        original=str(carla.WeatherParameters.ClearNoon)
        sunny=make_weather('day_sunny');cloudy=make_weather('day_cloudy');rain=make_weather('day_rainy')
        self.assertEqual(sunny.cloudiness,0.)
        self.assertEqual(sunny.fog_density,0.)
        self.assertGreater(cloudy.cloudiness,sunny.cloudiness)
        self.assertEqual(cloudy.precipitation,0.)
        self.assertEqual(rain.precipitation,90.)
        self.assertEqual(rain.wetness,100.)
        self.assertLess(make_weather('night_sunny').sun_altitude_angle,0.)
        self.assertGreater(make_weather('sunset_sunny').sun_altitude_angle,0.)
        self.assertEqual(str(carla.WeatherParameters.ClearNoon),original)

    def test_lamps_follow_weather_and_actual_braking(self):
        received=[]
        actor=NS(set_light_state=received.append)
        vehicle_lights(actor,'day_sunny')
        self.assertEqual(int(received[-1]),0)
        vehicle_lights(actor,'night_sunny',True)
        self.assertTrue(received[-1]&carla.VehicleLightState.LowBeam)
        self.assertTrue(received[-1]&carla.VehicleLightState.Position)
        self.assertTrue(received[-1]&carla.VehicleLightState.Brake)
        vehicle_lights(actor,'night_sunny',False)
        self.assertFalse(received[-1]&carla.VehicleLightState.Brake)
        vehicle_lights(actor,'day_rainy')
        self.assertTrue(received[-1]&carla.VehicleLightState.LowBeam)

    def test_particle_following_does_not_add_a_world_tick(self):
        class Base:
            ego=True
            def follow_ego(self):self.followed=True
            def tick(self):self.ticks+=1;return 12
        cls=with_straight_weather(Base)
        env=cls.__new__(cls)
        env.ticks=0
        self.assertEqual(env.tick(),12)
        self.assertTrue(env.followed)
        self.assertEqual(env.ticks,1)

    def test_camera_profile_is_condition_independent(self):
        self.assertEqual(CAMERA_ATTRIBUTES['exposure_mode'],'histogram')
        self.assertLess(float(CAMERA_ATTRIBUTES['exposure_min_bright']),10.)
        self.assertGreaterEqual(float(CAMERA_ATTRIBUTES['exposure_max_bright']),12.)


if __name__=='__main__':unittest.main()

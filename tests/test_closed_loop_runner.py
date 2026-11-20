"""Harness scenario and geometry contract checks without a CARLA server."""
import unittest
from types import SimpleNamespace as NS

from training.evaluate_closed_loop import LiveScenario, make_scenarios


class Waypoint:
    def __init__(self, x):
        self.road_id, self.section_id, self.lane_id, self.s = 1, 0, 1, x
        self.is_junction = False
        self.transform = NS(location=NS(x=x, y=0., z=0.), rotation=NS(pitch=0., yaw=0., roll=0.))

    def next(self, distance):
        return [Waypoint(self.s + distance)]


class RunnerTests(unittest.TestCase):
    def test_saved_scenarios_repeat_and_preserve_ten_metre_reset(self):
        world = NS(get_map=lambda: NS(generate_waypoints=lambda _: [Waypoint(0.), Waypoint(100.)]))
        a = make_scenarios(world, 2, 23, 20)
        self.assertEqual(a, make_scenarios(world, 2, 23, 20))
        self.assertEqual(len(a), 2)
        for scenario in a:
            self.assertEqual(scenario['leader'][0]-scenario['ego'][0], 10.)
            self.assertEqual(scenario['ego'][2], .5)
            self.assertGreater(len(scenario['route']), 30)

    def test_sensor_cleanup_continues_if_an_actor_is_already_gone(self):
        stopped = []
        class Actor:
            def stop(self):
                stopped.append('stop')
            def destroy(self):
                stopped.append('destroy')
                raise RuntimeError('already gone')
        live = LiveScenario.__new__(LiveScenario)
        live.actors = [Actor(), Actor()]
        live.close()
        self.assertEqual(stopped, ['stop', 'destroy', 'stop', 'destroy'])
        self.assertEqual(live.actors, [])


if __name__ == '__main__':
    unittest.main()

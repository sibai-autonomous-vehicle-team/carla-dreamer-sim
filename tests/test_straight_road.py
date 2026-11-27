"""Controlled-road geometry and shared training/evaluation routing."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from training.straight_road import opendrive, make_scenarios, scenarios_for_args


class StraightRoadTests(unittest.TestCase):
    def world(self, length=1000):
        def waypoint(road, lane, distance):
            self.assertEqual((road, lane), (1, -1))
            return NS(transform=NS(location=NS(x=distance, y=1.8, z=0.),
                                   rotation=NS(pitch=0., yaw=0., roll=0.)))
        return NS(get_map=lambda: NS(to_opendrive=lambda: opendrive(length),
                                    get_waypoint_xodr=waypoint))

    def test_one_straight_lane_without_junctions_or_objects(self):
        root=ET.fromstring(opendrive())
        self.assertEqual(len(root.findall('road')), 1)
        self.assertEqual(len(root.findall('.//geometry/line')), 1)
        self.assertEqual(len(root.findall('.//lane[@type="driving"]')), 1)
        self.assertEqual(root.findall('junction')+root.findall('.//object'), [])
        for length in (399, float('inf'), float('nan')):
            with self.assertRaises(ValueError): opendrive(length)

    def test_same_geometry_across_seeds_and_extended_road_is_used(self):
        a,b=make_scenarios(self.world(),2,42,100)
        self.assertEqual(a['full_route'], b['full_route'])
        self.assertNotEqual(a['seed'],b['seed'])
        self.assertTrue(all(p[1]==1.8 and p[4]==0 for p in a['full_route']))
        with self.assertRaisesRegex(ValueError, 'too short'):
            make_scenarios(self.world(),1,42,400)
        extended=make_scenarios(self.world(2000),1,42,400)[0]
        self.assertGreater(extended['full_route'][-1][0], 1900)

    def test_town_default_preserved_and_straight_dispatch_is_shared(self):
        with patch('training.evaluate_closed_loop.make_scenarios',return_value=['town']) as town:
            self.assertEqual(scenarios_for_args(None,1,42,100,NS()),['town'])
            town.assert_called_once_with(None,1,42,100)
        args=NS(road_environment='straight')
        self.assertEqual(scenarios_for_args(self.world(),1,42,100,args)[0]['seed'],42)


if __name__=='__main__': unittest.main()

import carla

from .agents.navigation.global_route_planner import GlobalRoutePlanner
from .base_planner import BasePlanner


import math


class FixedEndingPlanner(BasePlanner):
    """Route planner for a given vehicle and a given ending"""

    def __init__(self, vehicle: carla.Actor, end: carla.Location, sampling_radius=0.8):
        super().__init__(vehicle)
        self.end = end
        self._route_xy = ()
        self._grp = GlobalRoutePlanner(self._map, sampling_resolution=sampling_radius)

    def init_route(self):
        global_route = self._grp.trace_route(self._vehicle.get_location(), self.end)
        for waypoint in global_route:
            self.add_waypoint(waypoint[0])
        assert self.get_waypoint_num() > 0, "Empty waypoint queue"
        self._route_xy = tuple((x, y) for x, y, _ in self.get_all_waypoints())

    def distance_to_route(self, location):
        """Shortest horizontal distance to the planned route, in meters."""
        if not self._route_xy:
            return 0.0

        x, y = location[:2]
        if len(self._route_xy) == 1:
            px, py = self._route_xy[0]
            return math.hypot(x - px, y - py)

        best_squared = float("inf")
        for (ax, ay), (bx, by) in zip(self._route_xy, self._route_xy[1:]):
            dx, dy = bx - ax, by - ay
            length_squared = dx * dx + dy * dy
            fraction = 0.0
            if length_squared > 0:
                projection = ((x - ax) * dx + (y - ay) * dy) / length_squared
                fraction = max(0.0, min(1.0, projection))
            offset_x = x - (ax + fraction * dx)
            offset_y = y - (ay + fraction * dy)
            best_squared = min(best_squared, offset_x ** 2 + offset_y ** 2)
        return math.sqrt(best_squared)

    def extend_route(self):
        pass

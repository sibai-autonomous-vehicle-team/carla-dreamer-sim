"""Fixed Town03 routes for the gap PID comparison."""
import math
from dataclasses import dataclass

import carla
import numpy as np

from car_dreamer.carla_wpt_env import CarlaWptEnv
from car_dreamer.toolkit import BasePlanner
from carla_wrapper import CarlaWrapper


@dataclass(frozen=True)
class Scenario:
    name: str
    road: int
    lane: int
    start_s: float
    length: int
    finish_m: float
    branch_roads: tuple[int, ...] = ()


SCENARIOS = (
    Scenario("straight_east", 19, 1, 49, 46, 30),
    Scenario("straight_west", 19, -1, 5, 44, 30),
    Scenario("bend", 67, -2, 40, 44, 27),
    Scenario("left_turn", 19, 1, 13, 61, 44, (1284, 29)),
    Scenario("right_turn", 19, 1, 13, 59, 42, (1301, 28)),
)


class SharedRoute:
    def __init__(self, road_map, scenario):
        self.scenario = scenario
        waypoint = road_map.get_waypoint_xodr(scenario.road, scenario.lane, scenario.start_s)
        if waypoint is None:
            raise ValueError(f"No Town03 start for {scenario.name}")
        self.waypoints = [waypoint]
        for _ in range(scenario.length):
            successors = waypoint.next(1.0)
            if len(successors) > 1:
                successors = [candidate for candidate in successors
                              if candidate.road_id in scenario.branch_roads]
            if len(successors) != 1:
                raise ValueError(f"Route {scenario.name} needs an explicit branch at {waypoint.s:.1f}")
            waypoint = successors[0]
            self.waypoints.append(waypoint)

        self.xy = np.array([(w.transform.location.x, w.transform.location.y)
                            for w in self.waypoints])
        self.segments = np.diff(self.xy, axis=0)
        self.lengths = np.linalg.norm(self.segments, axis=1)
        self.cumulative = np.r_[0.0, self.lengths.cumsum()]
        if (self.lengths < 0.01).any() or self.cumulative[-1] < scenario.finish_m + 12:
            raise ValueError(f"Route {scenario.name} has invalid spacing or insufficient lead runout")

    def locate(self, location):
        offset = np.array([location.x, location.y]) - self.xy[:-1]
        along = np.clip((offset * self.segments).sum(axis=1) / self.lengths ** 2, 0, 1)
        errors = np.linalg.norm(offset - along[:, None] * self.segments, axis=1)
        segment = int(errors.argmin())
        return float(self.cumulative[segment] + self.lengths[segment] * along[segment]), float(errors[segment])

    def steer(self, actor):
        transform = actor.get_transform()
        progress, _ = self.locate(transform.location)
        target = min(progress + 3.0, self.cumulative[-1])
        x = np.interp(target, self.cumulative, self.xy[:, 0])
        y = np.interp(target, self.cumulative, self.xy[:, 1])
        heading = math.degrees(math.atan2(y - transform.location.y, x - transform.location.x))
        error = math.remainder(heading - transform.rotation.yaw, 360)
        return float(np.clip(0.03 * error, -1, 1))


class RouteWaypoints(BasePlanner):
    def __init__(self, actor, waypoints):
        super().__init__(actor)
        self.waypoints = waypoints

    def init_route(self):
        for waypoint in self.waypoints:
            self.add_waypoint(waypoint)

    def extend_route(self):
        pass


class SharedRouteFollowing(CarlaWrapper):
    """Use route progress and lane geometry with the existing CARLA environment."""

    lead_start_index = 10

    def select_scenario(self, scenario):
        self.route = SharedRoute(self._world.carla_map, scenario)

    def on_reset(self):
        self.random_num = 0
        if not 0 < self.lead_start_index < len(self.route.waypoints):
            raise ValueError(f"Lead start index outside route {self.route.scenario.name}")
        actors = []
        for index in (0, self.lead_start_index):
            transform = self.route.waypoints[index].transform
            transform.location.z += 0.5
            actor = self._world.spawn_actor(transform=transform)
            if actor is None:
                raise RuntimeError(f"Failed to spawn actor for {self.route.scenario.name}")
            actors.append(actor)
        self.ego, self.nonego = actors
        self.ego_planner = RouteWaypoints(self.ego, self.route.waypoints)
        self.on_step()

    def on_step(self):
        CarlaWptEnv.on_step(self)

    def ego_steer_command(self):
        # The wrapper negates the steering component before applying it.
        return -self.route.steer(self.ego)

    def get_nonego_vehicle_control(self):
        velocity = self.nonego.get_velocity()
        return carla.VehicleControl(
            throttle=2 / 3 if math.hypot(velocity.x, velocity.y) < 2 else 0,
            steer=self.route.steer(self.nonego),
        )

    def route_measurements(self):
        ego_progress, ego_deviation = self.route.locate(self.ego.get_location())
        lead_progress, lead_deviation = self.route.locate(self.nonego.get_location())
        return dict(ego_progress_m=ego_progress, ego_route_error_m=ego_deviation,
                    lead_progress_m=lead_progress, lead_route_error_m=lead_deviation)

    def is_destination_reached(self):
        progress, deviation = self.route.locate(self.ego.get_location())
        return progress >= self.route.scenario.finish_m and deviation <= 1.0

    def get_terminal_conditions(self):
        flags = CarlaWptEnv.get_terminal_conditions(self)
        _, deviation = self.route.locate(self.ego.get_location())
        flags["out_of_lane"] = deviation > self._config.terminal.out_lane_thres
        flags["terminal_dist"] = self.get_distance() > self._config.terminal.terminal_dist
        return flags

    def reward(self):
        return CarlaWptEnv.reward(self)

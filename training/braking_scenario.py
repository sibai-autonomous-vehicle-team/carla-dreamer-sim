"""Route-following lateral control for the true-gap braking collector."""

import math

from .evaluate_closed_loop import LiveScenario


class InvalidBrakingScenario(RuntimeError):
    """The sampled setup or route cannot produce a valid following rollout."""


class RouteTracker:
    """Track one persisted lane route with speed-dependent pure pursuit."""

    def __init__(self, vehicle, route):
        self.vehicle, self.route = vehicle, route
        self.index = 0
        self.lengths = [0.]
        for first, second in zip(route, route[1:]):
            self.lengths.append(self.lengths[-1] + math.hypot(second[0] - first[0], second[1] - first[1]))
        physics = vehicle.get_physics_control()
        front, rear = physics.wheels[:2], physics.wheels[2:]
        self.wheelbase = abs(sum(w.position.x for w in front) / len(front)
                             - sum(w.position.x for w in rear) / len(rear)) / 100.
        self.max_angle = math.radians(max(w.max_steer_angle for w in front))
        self.steering_curve = sorted((point.x, point.y) for point in physics.steering_curve)
        if self.wheelbase <= 0 or self.max_angle <= 0:
            raise ValueError('Vehicle geometry cannot support route tracking')

    def project(self, location):
        candidates = []
        for index in range(max(0, self.index - 3), min(len(self.route) - 1, self.index + 30)):
            first, second = self.route[index:index + 2]
            dx, dy = second[0] - first[0], second[1] - first[1]
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            along = max(0., min(1., ((location.x - first[0]) * dx + (location.y - first[1]) * dy) / length**2))
            x, y = first[0] + along * dx, first[1] + along * dy
            distance = math.hypot(location.x - x, location.y - y)
            candidates.append((distance, index, self.lengths[index] + along * length))
        if not candidates:
            raise RuntimeError('Persisted lane route has no usable segment')
        distance, self.index, progress = min(candidates)
        return progress, distance

    def point(self, progress):
        if not 0 <= progress < self.lengths[-1]:
            raise InvalidBrakingScenario('Vehicle reached the end of its persisted lane route')
        index = next(i for i in range(len(self.lengths) - 1) if self.lengths[i + 1] > progress)
        first, second = self.route[index:index + 2]
        weight = (progress - self.lengths[index]) / (self.lengths[index + 1] - self.lengths[index])
        point = [a + weight * (b - a) for a, b in zip(first[:3], second[:3])]
        yaw = math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))
        return point, yaw

    def steering(self):
        transform = self.vehicle.get_transform()
        speed = self.vehicle.get_velocity().length()
        progress, _ = self.project(transform.location)
        lookahead = max(4., .65 * speed)
        target, _ = self.point(progress + lookahead)
        heading = math.atan2(target[1] - transform.location.y, target[0] - transform.location.x)
        error = (heading - math.radians(transform.rotation.yaw) + math.pi) % (2 * math.pi) - math.pi
        distance = math.hypot(target[0] - transform.location.x, target[1] - transform.location.y)
        angle = math.atan2(2 * self.wheelbase * math.sin(error), max(distance, .1))
        # The vehicle movement component scales maximum steer by speed in km/h.
        scale = self.steering_scale(speed * 3.6)
        return max(-1., min(1., angle / (self.max_angle * scale)))

    def steering_scale(self, speed_kmh):
        if not self.steering_curve:
            return 1.
        if speed_kmh <= self.steering_curve[0][0]:
            return max(.01, self.steering_curve[0][1])
        for first, second in zip(self.steering_curve, self.steering_curve[1:]):
            if speed_kmh <= second[0]:
                weight = (speed_kmh - first[0]) / (second[0] - first[0])
                return max(.01, first[1] + weight * (second[1] - first[1]))
        return max(.01, self.steering_curve[-1][1])


class BrakingScenario(LiveScenario):
    """Keep both vehicles on the sampled route during setup and rollout."""

    def __init__(self, client, args, dt, scenario, **kwargs):
        self.lane_route = scenario['full_route']
        self.trackers = {}
        self.leader_events = []
        super().__init__(client, args, dt, scenario, **kwargs)
        try:
            self.validate_lanes()
            sensor = self.world.spawn_actor(self.world.get_blueprint_library().find('sensor.other.collision'),
                                            self.carla.Transform(), attach_to=self.nonego)
            self.actors.append(sensor)
            sensor.listen(lambda event: self.leader_events.append((event.frame, event.other_actor.id)))
        except BaseException:
            self.close()
            raise

    def tracker(self, actor):
        if actor.id not in self.trackers:
            self.trackers[actor.id] = RouteTracker(actor, self.lane_route)
        return self.trackers[actor.id]

    def place_ego(self, gap):
        """Adjust separation along the lane centerline, including curved spawns."""
        tracker = self.tracker(self.ego)
        leader_progress, _ = self.tracker(self.nonego).project(self.nonego.get_location())
        separation = gap + self.ego.bounding_box.extent.x + self.nonego.bounding_box.extent.x
        progress = leader_progress - separation
        for _ in range(8):
            point, yaw = tracker.point(progress)
            transform = self.ego.get_transform()
            old_waypoint = self.map.get_waypoint(transform.location)
            height = transform.location.z - old_waypoint.transform.location.z
            transform.location = self.carla.Location(x=point[0], y=point[1], z=point[2] + height)
            transform.rotation = self.carla.Rotation(pitch=self.map.get_waypoint(transform.location).transform.rotation.pitch,
                                                     yaw=yaw)
            error = self.get_distance(transform, self.nonego.get_transform()) - gap
            if abs(error) < .02:
                self.ego.set_transform(transform)
                return
            progress += error
        raise InvalidBrakingScenario('Could not place ego at the requested gap along the lane')

    def set_initial_gap(self, gap_target):
        for _ in range(4):
            self.place_ego(gap_target)
            for _ in range(20):
                self.world.tick()
            if abs(self.get_distance() - gap_target) <= .05:
                return
        raise InvalidBrakingScenario('Could not settle the initial lane-aligned bumper gap')

    def initialize_speed(self, speed):
        actors = (self.ego, self.nonego)
        commands = [speed, speed]
        for _ in range(math.ceil(3 / self.dt)):
            self.validate_lanes()
            for actor, command in zip(actors, commands):
                actor.apply_control(self.carla.VehicleControl(steer=self.tracker(actor).steering()))
                forward = actor.get_transform().get_forward_vector()
                actor.set_target_velocity(self.carla.Vector3D(
                    x=command * forward.x, y=command * forward.y, z=command * forward.z))
            self.world.tick()
            measured = [actor.get_velocity().length() for actor in actors]
            if all(abs(value - speed) <= .05 for value in measured):
                self.validate_lanes()
                return
            commands = [max(0., min(speed + 3., command + .5 * (speed - value)))
                        for command, value in zip(commands, measured)]
        raise InvalidBrakingScenario(f'Vehicles did not settle at {speed:.3f} m/s, measured speeds {measured}')

    def validate_handoff_speed(self):
        bounds = self.args.handoff_speed_bounds
        measured = [actor.get_velocity().length() for actor in (self.ego, self.nonego)]
        if any(not bounds[0] <= speed <= bounds[1] for speed in measured):
            raise InvalidBrakingScenario(f'Measured starting speeds {measured} outside requested range {bounds}')

    def set_moving_gap(self, bounds):
        for _ in range(5):
            if bounds[0] <= self.get_distance() <= bounds[1]:
                if self.events:
                    raise InvalidBrakingScenario('Contact occurred during moving-start initialization')
                self.validate_lanes()
                return
            speed = self.ego.get_velocity().length()
            self.place_ego(sum(bounds) / 2)
            forward = self.ego.get_transform().get_forward_vector()
            self.ego.set_target_velocity(self.carla.Vector3D(
                x=speed * forward.x, y=speed * forward.y, z=speed * forward.z))
            self.ego.apply_control(self.carla.VehicleControl(steer=self.tracker(self.ego).steering()))
            self.nonego.apply_control(self.carla.VehicleControl(steer=self.tracker(self.nonego).steering()))
            self.tick()
        raise InvalidBrakingScenario('Could not establish the requested moving bumper gap')

    def validate_lanes(self):
        offsets = []
        for name, actor in (('ego', self.ego), ('leader', self.nonego)):
            tracker = self.tracker(actor)
            progress, offset = tracker.project(actor.get_location())
            _, route_yaw = tracker.point(progress)
            heading_error = math.radians(actor.get_transform().rotation.yaw - route_yaw)
            half_width = (abs(math.cos(heading_error)) * actor.bounding_box.extent.y
                          + abs(math.sin(heading_error)) * actor.bounding_box.extent.x)
            waypoint = self.map.get_waypoint(actor.get_location(), lane_type=self.carla.LaneType.Driving)
            allowance = waypoint.lane_width / 2 - half_width + .15
            if offset > allowance:
                raise InvalidBrakingScenario(f'{name} left the planned lane, center offset {offset:.2f} m, '
                                             f'allowed offset {allowance:.2f} m, '
                                             f'heading error {math.degrees(heading_error):.1f} degrees')
            offsets.append(offset)
        return offsets

    def nominal_action(self, pid):
        return pid.update(float(self.get_distance()), self.dt), -self.tracker(self.ego).steering()

    def leader_control(self):
        steer = self.tracker(self.nonego).steering()
        elapsed = (self.world.get_snapshot().frame - self.initial_frame) * self.dt
        braking = self.args.leader_brake_after <= elapsed < self.args.leader_brake_after + self.args.leader_brake_duration
        if braking:
            return self.carla.VehicleControl(brake=1., steer=steer)
        speed = self.nonego.get_velocity().length()
        return self.carla.VehicleControl(throttle=2/3 if speed < self.args.initial_speed else 0., steer=steer)

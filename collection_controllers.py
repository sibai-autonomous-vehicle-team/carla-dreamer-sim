"""Additional collection policies; the original environment controller is unchanged."""
import numpy as np


class FullRangeFollowingController:
    """Distance/relative-speed feedback with temporally held full-range exploration.

    Commands are [acceleration command, steering command], not measured physical
    acceleration. Every five steps, choose uniform exploration with probability
    0.35; otherwise use feedback with small held noise. No original-controller
    actions are used. This is an exploration policy, not a safety controller.
    """

    def __init__(self, seed=42, exploration_probability=0.35, hold_steps=5):
        if not 0 <= exploration_probability <= 1 or hold_steps < 1:
            raise ValueError('Invalid exploration probability or hold duration')
        self.rng = np.random.default_rng(seed)
        self.exploration_probability = exploration_probability
        self.hold_steps = hold_steps
        self.reset()

    def reset(self):
        self.steps_remaining = 0
        self.exploring = False
        self.perturbation = np.zeros(2)

    @staticmethod
    def feedback(ego_position, ego_yaw, ego_velocity, leader_position, leader_velocity):
        displacement = np.asarray(leader_position)[:2] - np.asarray(ego_position)[:2]
        distance = np.linalg.norm(displacement)
        heading = np.deg2rad(ego_yaw)
        direction = displacement / distance if distance > 1e-6 else np.array([np.cos(heading), np.sin(heading)])
        relative_speed = np.dot(np.asarray(leader_velocity)[:2] - np.asarray(ego_velocity)[:2], direction)
        desired_heading = np.arctan2(displacement[1], displacement[0]) if distance > 1e-6 else heading
        heading_error = (desired_heading - heading + np.pi) % (2 * np.pi) - np.pi
        return np.array([0.5 * (distance - 8.0) + 0.8 * relative_speed, -0.75 * heading_error])

    def __call__(self, env):
        ego, leader = env.ego, env.nonego
        ego_tf, leader_tf = ego.get_transform(), leader.get_transform()
        ego_v, leader_v = ego.get_velocity(), leader.get_velocity()
        feedback = self.feedback(
            [ego_tf.location.x, ego_tf.location.y], ego_tf.rotation.yaw,
            [ego_v.x, ego_v.y], [leader_tf.location.x, leader_tf.location.y],
            [leader_v.x, leader_v.y],
        )
        if self.steps_remaining == 0:
            self.exploring = self.rng.random() < self.exploration_probability
            self.perturbation = (
                self.rng.uniform([-3.0, -1.0], [3.0, 1.0]) if self.exploring
                else self.rng.normal(0, [0.3, 0.05])
            )
            self.steps_remaining = self.hold_steps
        self.steps_remaining -= 1
        action = self.perturbation if self.exploring else feedback + self.perturbation
        return np.clip(action, [-3.0, -1.0], [3.0, 1.0]).astype(np.float32)

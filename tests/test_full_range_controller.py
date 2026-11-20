from types import SimpleNamespace
import unittest

import numpy as np

from collection_controllers import FullRangeFollowingController


def actor(x, speed):
    return SimpleNamespace(
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=x, y=0), rotation=SimpleNamespace(yaw=0),
        ),
        get_velocity=lambda: SimpleNamespace(x=speed, y=0),
    )


class FullRangeControllerTest(unittest.TestCase):
    def test_feedback_brakes_when_close_or_closing_and_accelerates_when_far(self):
        feedback = FullRangeFollowingController.feedback
        self.assertLess(feedback([0, 0], 0, [2, 0], [4, 0], [2, 0])[0], 0)
        self.assertLess(feedback([0, 0], 0, [12, 0], [12, 0], [2, 0])[0], 0)
        self.assertGreater(feedback([0, 0], 0, [2, 0], [20, 0], [2, 0])[0], 0)
        np.testing.assert_allclose(feedback([0, 0], 0, [2, 0], [8, 0], [2, 0]), [0, 0])

    def test_exploration_bounds_coverage_and_reproducibility(self):
        env = SimpleNamespace(ego=actor(0, 2), nonego=actor(8, 2))
        policies = [FullRangeFollowingController(seed=42) for _ in range(2)]
        actions = np.array([policies[0](env) for _ in range(10000)])
        repeated = np.array([policies[1](env) for _ in range(10000)])
        np.testing.assert_array_equal(actions, repeated)
        self.assertEqual(actions.dtype, np.float32)
        self.assertTrue(np.isfinite(actions).all())
        self.assertTrue((actions >= [-3, -1]).all() and (actions <= [3, 1]).all())
        self.assertLess(actions[:, 0].min(), -2.9)
        self.assertGreater(actions[:, 0].max(), 2.9)
        self.assertLess(actions[:, 1].min(), -0.98)
        self.assertGreater(actions[:, 1].max(), 0.98)

    def test_held_exploration_and_reset(self):
        env = SimpleNamespace(ego=actor(0, 2), nonego=actor(8, 2))
        policy = FullRangeFollowingController(exploration_probability=1, hold_steps=5)
        actions = [policy(env) for _ in range(5)]
        for action in actions[1:]:
            np.testing.assert_array_equal(action, actions[0])
        policy.reset()
        self.assertEqual(policy.steps_remaining, 0)
        self.assertFalse(np.array_equal(policy(env), actions[0]))


if __name__ == '__main__':
    unittest.main()

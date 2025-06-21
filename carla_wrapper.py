from car_dreamer.carla_follow_env import CarlaFollowEnv
import numpy as np

class CarlaWrapper(CarlaFollowEnv):
    def __init__(self, config):
        super().__init__(config)

    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        init_state = self.reset()
        goal_state = self.reset()

    def reset(self, reset_to_state=None):
        """
        Reset the environment and return the initial observation.
        """
        self.reset_to_state = reset_to_state
        super().reset()
        return self.state()
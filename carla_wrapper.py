from car_dreamer.carla_follow_env import CarlaFollowEnv
import numpy as np

class CarlaWrapper(CarlaFollowEnv):
    def __init__(self, config):
        super().__init__(config)
        # self.action_space = self._get_action_space().

    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        _, init_state = self.reset()
        _, goal_state = self.reset()
        return init_state, goal_state

    def reset(self, reset_to_state=None):
        """
        Reset the environment and return the initial observation.
        """
        self.reset_to_state = reset_to_state
        obs = super().reset()
        return obs, self.state()
    
    def update_env(self, env_info):
        """
        Update the environment with the given information.
        This method can be used to update the environment state or configuration.
        """
        self.shape = env_info['shape']
    
    def eval_state(self, goal_state, cur_state):
        """
        Evaluate the current state against the goal state.
        This method can be used to determine how close the current state is to the goal state.
        """
        # Example evaluation: return the Euclidean distance between the two states
        cur_ego_pos = np.array(cur_state[0][:2])
        cur_nonego_pos = np.array(cur_state[1][:2])
        cur = np.concatenate((cur_ego_pos, cur_nonego_pos))

        goal_ego_pos = np.array(goal_state[0][:2])
        goal_nonego_pos = np.array(goal_state[1][:2])
        goal = np.concatenate((goal_ego_pos, goal_nonego_pos))

        pos_diff = np.linalg.norm(cur - goal)

        success = pos_diff < 1.0  # Define success condition based on distance threshold
        state_dist = np.linalg.norm(cur_state - goal_state)

        return {
            'success': success,
            'state_dist': state_dist
        }
    
    def prepare(self, seed, init_state):
        """
        Prepare the environment with a specific seed and initial state.
        This method can be used to set up the environment before starting an episode.
        """
        self.seed(seed)
        return self.reset(init_state)
    
    

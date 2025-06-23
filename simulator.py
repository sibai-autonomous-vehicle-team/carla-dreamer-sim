import gym
import car_dreamer
from carla_wrapper import get_image_obs
import numpy as np
from carla_wrapper import CarlaWrapper
from gym.envs.registration import register

# Initialize the environment
config = car_dreamer.load_task_configs('carla_follow')
# env = gym.make('CarlaFollowEnv-v0', config=config.env)

register(
    id='CarlaFollowWrapperEnv-v0',
    entry_point='carla_wrapper:CarlaWrapper',
)

env = gym.make('CarlaFollowWrapperEnv-v0', config=config.env)

reset_to_state = [
    [-10.856332778930664, -135.0041046142578, -0.008075752295553684, 0.06346611678600311, 1.2272632122039795, -5.211023626494615e-11], 
    [-0.8586273193359375, -134.7899169921875, -0.008075752295553684, 0.06346611678600311, 1.2272632122039795, -5.211023626494615e-11]
]

observation, state = env.reset(reset_to_state=reset_to_state)

episode_count = 0
step_index = 0
dataset = {
    'states': [],  # list of states
    'observations': [],  # list of obs
    'actions': [],
    'costs': []
}


print('Collect Initial Data')
i = 1
terminated = False
# Collect data until the episode is terminated
while terminated is False:
    action = env.compute_continuous_action() # Sample a random action
    next_obs, reward, terminated, info = env.step(action)

    dataset['states'].append(np.array(env.state()))
    dataset['observations'].append(np.array(get_image_obs(observation)))
    dataset['actions'].append(action)
    dataset['costs'].append(int(env.is_collision()))

    print(f"Step {i} - Reward: {reward}, Cost: {int(env.is_collision())} Terminated: {terminated} Action: {action} eval_state: {env.eval_state(reset_to_state, env.state())}")
    observation = next_obs
    i += 1


env.rollout(42, init_state=dataset['states'][0], actions=dataset['actions'])
   


env.close()


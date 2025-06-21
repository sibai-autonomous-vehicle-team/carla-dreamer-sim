import carla
import gym
import yaml
import car_dreamer
from car_dreamer.carla_follow_env import CarlaFollowEnv


# Initialize the environment
config = car_dreamer.load_task_configs('carla_follow')
env = gym.make('CarlaFollowEnv-v0', config=config.env)

# env.reset_to_state = [
#     [-10.856332778930664, -135.0041046142578, -0.008075752295553684, 0.06346611678600311, 1.2272632122039795, -5.211023626494615e-11], 
#     [-0.8586273193359375, -134.7899169921875, -0.008075752295553684, 0.06346611678600311, 1.2272632122039795, -5.211023626494615e-11]
# ]

observation = env.reset()

episode_count = 0
step_index = 0
dataset = {
    'states': [],  # list of states
    'observations': [],  # list of obs
    'actions': [],
    'cost': []
}


print('Collect Initial Data')
for i in range(500):
    action = env.compute_continuous_action() # Sample a random action
    next_obs, reward, terminated, info = env.step(action)

    dataset['states'].append(env.state())
    dataset['observations'].append(observation)
    dataset['actions'].append(action)
    dataset['cost'].append(int(env.is_collision()))

    print(f"Step {i} - Reward: {reward}, Cost: {int(env.is_collision())} Terminated: {terminated} Action: {action}")
    observation = next_obs
    if terminated:
        obs = env.reset()
print(dataset['states'][0], dataset['states'][1])
print(len(dataset['states']), len(dataset['observations']), len(dataset['actions']), len(dataset['cost']))

env.close()


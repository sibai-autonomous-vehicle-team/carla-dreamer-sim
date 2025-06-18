import carla
import gym
import yaml
import car_dreamer
from car_dreamer.carla_follow_env import CarlaFollowEnv


# Initialize the environment
config = car_dreamer.load_task_configs('carla_follow')
env = gym.make('CarlaFollowEnv-v0', config=config.env)
observation = env.reset()


episode_count = 0
step_index = 0
dataset = {
    'states': [...],  # list of states
    'observations': [...],  # list of obs
    'actions': [...],
    'cost': [...]
}


print('Collect Initial Data')
for i in range(35):
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

env.close()


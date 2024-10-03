import gym
import pygame
from matplotlib import pyplot as plt

from maze_v11 import MazeEnv

# Register the custom environment
gym.register(
    id='Maze_v11',
    entry_point='maze_v11:MazeEnv',
    kwargs={'maze': None}
)

# Define the maze
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'S', 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 'G', 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 'E', 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
#
# episodes = 10
# sleep_sec = 0
# # sleep_sec = 0.05
#
# # Initialize the environment
# env_q = gym.make('Maze_v11', maze=maze, algorithm="q-learning")
# obs_q = env_q.reset()
# env_q.render()
# rewards_q = env_q.train(episodes, sleep_sec)
#
# env_sarsa = gym.make('Maze_v11', maze=maze, algorithm="sarsa") # TODO same env?
# obs_sarsa = env_sarsa.reset()
# env_sarsa.render()
# rewards_sarsa = env_sarsa.train(episodes, sleep_sec)
#
# # env_policy_gradient = gym.make('Maze_v11', maze=maze, algorithm="policy_gradient") # TODO same env?
# # obs_policy_gradient = env_policy_gradient.reset()
# # env_policy_gradient.render()
# # rewards_policy_gradient = env_policy_gradient.train(episodes, sleep_sec)
#
# # env_random = gym.make('Maze_v11', maze=maze, algorithm="random") # TODO same env?
# # obs_random = env_random.reset()
# # env_random.render()
# # rewards_random = env_random.train(episodes, sleep_sec)
# #
# # # Visualize the learned path after training
# # env.visualize_learned_path() TODO
#
#
# # Plotting the rewards over episodes for comparison
# plt.plot(rewards_q, label='Q-learning')
# plt.plot(rewards_sarsa, label='SARSA')
# # plt.plot(rewards_policy_gradient, label='policy_gradient')
# # plt.plot(rewards_random, label='random')
# plt.xlabel('Episodes')
# plt.ylabel('Total Rewards')
# plt.legend()
# plt.show()

############################################
import numpy as np
import matplotlib.pyplot as plt
import gym

# Parameters
episodes = 10
sleep_sec = 0
num_experiments = 2
algorithms = ["q-learning", "sarsa"]  # Add more algorithms as needed: "policy_gradient", "random"



# Initialize a dictionary to store the results for each algorithm
results = {alg: [] for alg in algorithms}

# Run the experiments for each algorithm 100 times
for alg in algorithms:
    rewards = []
    for experiment in range(1, num_experiments+1):
        # Create environment for the current algorithm
        env = gym.make('Maze_v11', maze=maze, algorithm=alg, experiment=experiment)
        env.reset()
        env.render()
        rewards.append(env.train(episodes, sleep_sec))
        # return np.array(rewards)
        # rewards = run_experiment(env, episodes, sleep_sec)
        results[alg].append(rewards)

print(results)
# Convert lists to numpy arrays for easy computation
for alg in algorithms:
    results[alg] = np.array(results[alg])

# Calculate the average and standard deviation of rewards across the experiments
mean_rewards = {}
std_rewards = {}

for alg in algorithms:
    mean_rewards[alg] = np.mean(results[alg], axis=0)
    std_rewards[alg] = np.std(results[alg], axis=0)

# Plot the results with error bars
for alg in algorithms:
    plt.errorbar(range(episodes), mean_rewards[alg], yerr=std_rewards[alg], label=alg, capsize=5)

plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.legend()
plt.title('Average Total Rewards with Standard Deviations')
plt.show()

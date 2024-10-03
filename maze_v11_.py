import gym
import pygame
from matplotlib import pyplot as plt
import numpy as np
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

# Parameters
episodes = 20
sleep_sec = 0
num_experiments = 100
show = False
algorithms = ["q-learning", "sarsa"]  # Add more algorithms as needed: "policy_gradient", "random"

# Initialize a dictionary to store the results for each algorithm
results = {alg: [] for alg in algorithms}

# Run the experiments for each algorithm 100 times
for alg in algorithms:
    for experiment in range(1, num_experiments+1):
        rewards = []
        # Create environment for the current algorithm
        env = gym.make('Maze_v11', maze=maze, algorithm=alg, experiment=experiment, show=show)
        env.reset()
        env.render()
        # rewards.append(env.train(episodes, sleep_sec))
        # return np.array(rewards)
        # rewards = run_experiment(env, episodes, sleep_sec)
        # Collect rewards over episodes for this experiment
        rewards = env.train(episodes, sleep_sec)
        results[alg].append(rewards)

# print(results)
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
# for alg in algorithms:
#     plt.errorbar(range(episodes), mean_rewards[alg], yerr=std_rewards[alg], label=alg, capsize=5)

for alg in algorithms:
    plt.plot(range(episodes), mean_rewards[alg], label=alg)
    plt.fill_between(range(episodes),
                     mean_rewards[alg] - std_rewards[alg],
                     mean_rewards[alg] + std_rewards[alg],
                     alpha=0.2)  # alpha controls transparency

plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.legend()
plt.title('Average Total Rewards with Standard Deviations')
plt.show()

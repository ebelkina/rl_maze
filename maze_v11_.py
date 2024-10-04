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
maze_10x10 = [
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
maze=maze_10x10
episodes = 10
sleep_sec = 0
num_experiments = 10
show = True
algorithms = ["q-learning", "sarsa"]  # Add more algorithms as needed: "policy_gradient", "random"
epsilons = [0, 0.1]

# Initialize a dictionary to store the results for each algorithm
results = {(alg, eps): [] for alg in algorithms for eps in epsilons}

# Run the experiments for each algorithm 100 times
for eps in epsilons:
    for alg in algorithms:
        for experiment in range(1, num_experiments+1):
            rewards = []
            # Create environment for the current algorithm
            env = gym.make('Maze_v11', maze=maze, epsilon=eps, algorithm=alg, experiment=experiment, show=show)
            env.reset()
            env.render()
            # rewards.append(env.train(episodes, sleep_sec))
            # return np.array(rewards)
            # rewards = run_experiment(env, episodes, sleep_sec)
            # Collect rewards over episodes for this experiment
            rewards = env.train(episodes, sleep_sec)
            # Append rewards for this experiment to the results dictionary
            results[(alg, eps)].append(rewards)

# print(results)
# Convert lists to numpy arrays for easy computation
for key in results:
    results[key] = np.array(results[key])

# Calculate the average and standard deviation of rewards across the experiments
mean_rewards = {}
std_rewards = {}

for key in results:
    mean_rewards[key] = np.mean(results[key], axis=0)
    std_rewards[key] = np.std(results[key], axis=0)

# Plot the results with shaded areas for standard deviation
plt.figure()

for eps in epsilons:
    for alg in algorithms:
        label = f"{alg} {eps})"
        plt.plot(range(episodes), mean_rewards[(alg, eps)], label=label)
        plt.fill_between(range(episodes),
                         mean_rewards[(alg, eps)] - std_rewards[(alg, eps)],
                         mean_rewards[(alg, eps)] + std_rewards[(alg, eps)],
                         alpha=0.2)  # alpha controls transparency

plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.legend(loc='lower right')
plt.title('Average Total Rewards with Standard Deviations for Different Epsilons')
plt.grid(which='major', linestyle='--', linewidth=0.75)
plt.show()

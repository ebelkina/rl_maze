import gym
import pygame
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from maze_v17 import MazeEnv
from scipy.stats import ttest_rel

# Register the custom environment
gym.register(
    id='Maze_v17',
    entry_point='maze_v17:MazeEnv',
    kwargs={'maze': None}
)

# Define the maze
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'S', 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 'E', 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

maze_np = np.array(maze)

# Set sub-gaol
# maze[5][3] = 'G' # initial
# maze[7][3] = 'G'
# maze[2][3] = 'G'
# maze[1][3] = 'G'
maze[1][4] = 'G'
# maze[6][8] = 'G'

# Parameters
episodes = 30

sleep_sec = 0
# sleep_sec = 0.05
# sleep_sec = 0.5

num_experiments = 1 # equal seed
show = False
show = True

# show = False
algorithms = ["q-learning"]
# algorithms = ["q-learning", "sarsa"]

epsilons = [0.001]
# epsilons = [0, 0.001, 0.01, 0.1]

reduce_epsilon = False
reduce_epsilon = True

alpha = 0.05

# Initialize a dictionary to store the results for each algorithm
# keys: ('q-learning', 0), ('q-learning', 0.1), ('sarsa', 0), ('sarsa', 0.1)
results = {(alg, eps): [] for alg in algorithms for eps in epsilons}

# Maximum possible reward for the maze
max_possible_reward = 137  # TODO check

# Run the experiments
for eps in epsilons:
    for alg in algorithms:
        for experiment in range(1, num_experiments+1):
            rewards = []
            env = gym.make('Maze_v17', maze=maze, epsilon=eps, algorithm=alg,
                           experiment=experiment, show=show, reduce_epsilon=reduce_epsilon)
            env.reset()
            env.render()
            rewards, q_table_1, q_table_2 = env.train(episodes, sleep_sec)
            results[(alg, eps)].append(rewards)
            env.show_learned_path()

# Convert results to numpy arrays for easy computation
for key in results:  # keys: ('q-learning', 0), ('q-learning', 0.1), ('sarsa', 0), ('sarsa', 0.1)
    results[key] = np.array(results[key])
    # print(key)
    # print(results[key])

# Calculate average and standard deviation of rewards
mean_total_rewards = {}
std_total_rewards = {}
for key in results: # keys: ('q-learning', 0), ('q-learning', 0.1), ('sarsa', 0), ('sarsa', 0.1)
    mean_total_rewards[key] = np.mean(results[key], axis=0)
    std_total_rewards[key] = np.std(results[key], axis=0)


# Paired t-test between Q-learning and SARSA with epsilon = 0.1
# q_learning_rewards = mean_rewards[('q-learning', 0.1)]
# sarsa_rewards = mean_rewards[('sarsa', 0.1)]
# print("q_learning_rewards", q_learning_rewards)
# print("sarsa_rewards", sarsa_rewards)
# t_stat, p_value = ttest_rel(q_learning_rewards, sarsa_rewards)
# print(p_value)
# print(f"Paired t-test between Q-learning (epsilon = 0.1) and SARSA (epsilon = 0.1):")
# print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
#
# if p_value < alpha:
#     print("Conclusion: There is a statistically significant difference between the rewards of Q-learning and SARSA with epsilon = 0.1 (p < 0.05).")
# else:
#     print("Conclusion: There is no statistically significant difference between the rewards of Q-learning and SARSA with epsilon = 0.1 (p >= 0.05).")
#
# Calculate first episode where the algorithms reach maximum possible reward
first_reach_max = {alg: None for alg in algorithms}
num_not_reach_max = {alg: 0 for alg in algorithms}

for alg in algorithms:
    alg_rewards = np.mean(results[(alg, 0)], axis=0)
    for episode in range(episodes):
        if alg_rewards[episode] >= max_possible_reward and first_reach_max[alg] is None:
            first_reach_max[alg] = episode
        if alg_rewards[episode] < max_possible_reward:
            num_not_reach_max[alg] += 1

# Save the results to CSV
df_results = pd.DataFrame({f"{alg}_{eps}": np.mean(results[(alg, eps)], axis=0) for alg in algorithms for eps in epsilons})
df_results.to_csv('results.csv', index=False)

# Print maximum reward comparison stats
print("First episode where Q-learning reached max reward:", first_reach_max['q-learning'])
# print("First episode where SARSA reached max reward:", first_reach_max['sarsa'])
print("Number of times Q-learning did not reach max reward:", num_not_reach_max['q-learning'])
# print("Number of times SARSA did not reach max reward:", num_not_reach_max['sarsa'])

# Plot the results
plt.figure()
for eps in epsilons:
    for alg in algorithms:
        label = f"{alg}_{eps}"
        plt.plot(range(episodes), mean_total_rewards[(alg, eps)], label=label)
        plt.fill_between(range(episodes),
                         mean_total_rewards[(alg, eps)] - std_total_rewards[(alg, eps)],
                         mean_total_rewards[(alg, eps)] + std_total_rewards[(alg, eps)],
                         alpha=0.2)

plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.ylim(-200, 200)
plt.legend(loc='lower right')
plt.title('Average Total Rewards with Standard Deviations for Different Epsilons')
plt.grid(which='major', linestyle='--', linewidth=0.75)
plt.show()

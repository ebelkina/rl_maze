import gym
import pygame
from matplotlib import pyplot as plt

from maze_v10 import MazeEnv

# Register the custom environment
gym.register(
    id='Maze_v10',
    entry_point='maze_v10:MazeEnv',
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

# Initialize the environment using gym.make
env = gym.make('Maze_v10', maze=maze)
obs = env.reset()
env.render()
# env.train(episodes=100, sleep_sec=0.05)  # sleep_sec=0.05
q_rewards = env.train(episodes=10, algorithm="q-learning")#, sleep_sec=0.05)
sarsa_rewards = env.train(episodes=10, algorithm="sarsa")#, sleep_sec=0.05)
#
# # Visualize the learned path after training
# env.visualize_learned_path() TODO


# Plotting the rewards over episodes for comparison
plt.plot(q_rewards, label='Q-learning')
plt.plot(sarsa_rewards, label='SARSA')
plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.legend()
plt.show()
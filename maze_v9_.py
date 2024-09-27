import gym
import pygame
from maze_v9 import MazeEnv

# Register the custom environment
gym.register(
    id='Maze_v9',
    entry_point='maze_v9:MazeEnv',
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
env = gym.make('Maze_v9', maze=maze)
obs = env.reset()
env.render()
env.train(episodes=100)
#
# # Visualize the learned path after training
# env.visualize_learned_path()
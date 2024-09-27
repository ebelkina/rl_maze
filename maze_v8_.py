import gym
import pygame

from maze_v8 import MazeGameEnv

# Register the environment
gym.register(
    id='MazeGame-v0',
    entry_point='maze_v8:MazeGameEnv',
    kwargs={'maze': None}
)

#Maze config

# maze = [
#     ['S', '', '.', '.'],
#     ['.', '#', '.', '#'],
#     ['.', '.', '.', '.'],
#     ['#', '.', '#', 'G'],
# ]

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

# Test the environment
env = gym.make('MazeGame-v0',maze=maze)
obs = env.reset()
env.render()


def get_valid_actions(env):
    possible_actions = list(range(env.action_space.n))  # Get all possible actions from env.action_space
    valid_actions = []

    # Get the agent's current position
    current_pos = np.array(env.current_pos)

    # Check each action and see if it leads to a valid position
    for action in possible_actions:
        new_pos = np.array(current_pos)  # Start from the current position

        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

        # Check if the new position is valid
        if env._is_valid_position(new_pos):
            valid_actions.append(action)  # Add to the list of valid actions

done = False
while True:
    pygame.event.get()
    print("******************")
    action = env.action_space.sample()  # Random action selection
    # possible_actions = {1, 2, 3, 4}
    #
    # for action in list(possible_actions):  # Convert set to list for safe iteration
    #     obs, reward, done, _ = env.step(action)
    #     env.render()
    #     print('Reward:', reward)
    #     print('Done:', done)
    #
    #     # Remove the action from the set after using it
    #     possible_actions.discard(action)

    obs, reward, done, _, _ = env.step(action)
    env.render()
    print('Reward:', reward)
    print('Done:', done)

    pygame.time.wait(200)
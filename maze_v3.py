import numpy as np
import random

# Maze definition (1 = wall, 0 = open, S = start, G = sub-goal, E = end)
maze = np.array([
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
])

# Define start, sub-goal, and end
start = (1, 1)
sub_goal = (5, 3)
end_goal = (8, 7)

# Actions (Up, Down, Left, Right)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Q-Learning Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Epsilon-greedy parameter
episodes = 5000  # Number of episodes

# Initialize Q-table with zeros
q_table = np.zeros((maze.shape[0], maze.shape[1], len(actions)))


# Check if the action leads to a valid position in the maze
def is_valid_position(pos):
    return 0 <= pos[0] < maze.shape[0] and 0 <= pos[1] < maze.shape[1] and maze[pos] != 1


# Choose action using epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))  # Random action
    else:
        return np.argmax(q_table[state])  # Best action


# Perform the action and return new state and reward
def take_action(state, action):
    new_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    if not is_valid_position(new_state):
        return state, -1  # Penalty for hitting a wall

    # Reward settings
    if new_state == sub_goal:
        return new_state, 10  # Reward for reaching sub-goal
    if new_state == end_goal:
        return new_state, 100  # Reward for reaching final goal
    return new_state, -0.1  # Small penalty for each move


# Q-learning main loop
for episode in range(episodes):
    state = start
    reached_sub_goal = False

    while state != end_goal:
        action = choose_action(state)
        next_state, reward = take_action(state, action)

        if next_state == sub_goal:
            reached_sub_goal = True

        # Update Q-value
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += alpha * (
                    reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])

        state = next_state

        # If reached end goal, end the episode
        if state == end_goal:
            break

import matplotlib.pyplot as plt


def visualize_policy(q_table):
    policy = np.zeros((maze.shape[0], maze.shape[1]), dtype=str)

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 1:
                policy[i, j] = 'X'  # Wall
            elif maze[i, j] == 'S':
                policy[i, j] = 'S'  # Start
            elif maze[i, j] == 'G':
                policy[i, j] = 'G'  # Sub-goal
            elif maze[i, j] == 'E':
                policy[i, j] = 'E'  # End goal
            else:
                best_action = np.argmax(q_table[i, j])
                if best_action == 0:
                    policy[i, j] = '↑'
                elif best_action == 1:
                    policy[i, j] = '↓'
                elif best_action == 2:
                    policy[i, j] = '←'
                elif best_action == 3:
                    policy[i, j] = '→'

    print(policy)


visualize_policy(q_table)

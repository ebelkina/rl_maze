import numpy as np
import random

# Define the maze size
MAZE_SIZE = 10

# Define the actions: up, down, left, right
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_MAP = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}

class MazeEnv:
    def __init__(self, maze, start, sub_goal, final_goal):
        self.maze = maze
        self.start = start
        self.sub_goal = sub_goal
        self.final_goal = final_goal
        self.current_position = start
        self.visited_sub_goal = False

    def reset(self):
        """Reset the environment to the starting position and mark the sub-goal as not visited."""
        self.current_position = self.start
        self.visited_sub_goal = False
        return self.current_position

    def step(self, action):
        """Apply an action, return the new state, reward, and done flag."""
        # Map the action to position changes
        move = ACTION_MAP[action]
        new_position = (self.current_position[0] + move[0], self.current_position[1] + move[1])

        # Check if new position is out of bounds or a wall
        if (0 <= new_position[0] < MAZE_SIZE and
            0 <= new_position[1] < MAZE_SIZE and
            self.maze[new_position[0], new_position[1]] != 1):  # 1 means wall
            self.current_position = new_position
        else:
            # Invalid move, stay in the same position
            new_position = self.current_position

        # Check if the agent has reached the sub-goal
        if new_position == self.sub_goal:
            self.visited_sub_goal = True
            reward = 10  # Large reward for reaching sub-goal
        # Check if the agent has reached the final goal after the sub-goal
        elif new_position == self.final_goal and self.visited_sub_goal:
            reward = 100  # Large reward for reaching final goal
            done = True
        else:
            reward = -1  # Small negative reward to encourage faster solving
            done = False

        return new_position, reward, done

    def is_done(self):
        """Check if the agent has reached the final goal."""
        return self.current_position == self.final_goal and self.visited_sub_goal

    def render(self):
        """Visualize the current state of the maze."""
        maze_copy = self.maze.copy()
        maze_copy[self.current_position] = 2  # Mark current position of agent
        maze_copy[self.sub_goal] = 3  # Mark sub-goal
        maze_copy[self.final_goal] = 4  # Mark final goal
        print(maze_copy)

# Maze representation: 0 = free space, 1 = wall, 2 = start, 3 = sub-goal, 4 = final goal
maze = np.array([
    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
])

# Define start, sub-goal, and final goal positions
start = (0, 0)
sub_goal = (5, 5)
final_goal = (9, 9)

# Create the environment
env = MazeEnv(maze, start, sub_goal, final_goal)

# Reset environment and display it
env.reset()
env.render()


# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995
num_episodes = 1000
max_steps_per_episode = 100

# Initialize the Q-table: rows for states, columns for actions (4 actions)
q_table = np.zeros((MAZE_SIZE, MAZE_SIZE, len(ACTIONS)))

def choose_action(state):
    """Choose an action using the ε-greedy policy."""
    if np.random.random() < epsilon:
        return random.choice(ACTIONS)  # Explore
    else:
        state_q_values = q_table[state[0], state[1], :]
        return ACTIONS[np.argmax(state_q_values)]  # Exploit

def q_learning(env, q_table, num_episodes, max_steps):
    global epsilon  # Use the epsilon declared above

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0

        for step in range(max_steps):
            # Choose action
            action = choose_action(state)

            # Take action and observe new state, reward, and done flag
            next_state, reward, done = env.step(action)

            # Q-learning update
            action_index = ACTIONS.index(action)
            best_next_action_index = np.argmax(q_table[next_state[0], next_state[1], :])
            q_table[state[0], state[1], action_index] += alpha * (reward + gamma * q_table[next_state[0], next_state[1], best_next_action_index] - q_table[state[0], state[1], action_index])

            # Move to the next state
            state = next_state
            steps += 1

            if done:
                break

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % 100 == 0:
            print(f"Episode {episode}: completed in {steps} steps.")

    return q_table

# Run the Q-learning algorithm
trained_q_table = q_learning(env, q_table, num_episodes, max_steps_per_episode)

# Display final Q-table for visualization
print("Final Q-table:")
print(trained_q_table)

# def test_agent(env, q_table):
#     state = env.reset()
#     done = False
#     steps = 0
#
#     while not done and steps < max_steps_per_episode:
#         env.render()
#         action_index = np.argmax(q_table[state[0], state[1], :])
#         action = ACTIONS[action_index]
#
#         # Take the action and move to the next state
#         state, _, done = env.step(action)
#         steps += 1
#
#     print(f"Test completed in {steps} steps.")
#
# # Test the agent with the learned Q-values
# test_agent(env, trained_q_table)


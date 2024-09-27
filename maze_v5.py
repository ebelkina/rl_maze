import sys
import numpy as np
import math
import random

# A simple custom environment to replace gym_maze
class SimpleMazeEnv:
    def __init__(self, size=(10, 10)):
        self.size = size
        self.observation_space = self.ObservationSpace(size)
        self.action_space = self.ActionSpace()
        self.state = (0, 0)  # Starting position
        self.goal = (size[0] - 1, size[1] - 1)  # Goal position at the bottom-right corner

    class ObservationSpace:
        def __init__(self, size):
            self.low = np.zeros(len(size))
            self.high = np.array(size) - 1
            self.shape = len(size)

    class ActionSpace:
        def __init__(self):
            self.n = 4  # 4 possible actions: N, S, E, W

        def sample(self):
            return random.randint(0, 3)

    def reset(self):
        self.state = (0, 0)
        return np.array(self.state)

    def step(self, action):
        x, y = self.state
        if action == 0 and x > 0:  # North
            x -= 1
        elif action == 1 and x < self.size[0] - 1:  # South
            x += 1
        elif action == 2 and y < self.size[1] - 1:  # East
            y += 1
        elif action == 3 and y > 0:  # West
            y -= 1

        self.state = (x, y)
        reward = -1  # Penalty for every move
        done = self.state == self.goal  # Reached the goal
        if done:
            reward = 100  # Reward for reaching the goal

        return np.array(self.state), reward, done, {}

    def render(self):
        maze = np.zeros(self.size)
        x, y = self.state
        maze[x, y] = 1  # Current position
        gx, gy = self.goal
        maze[gx, gy] = 2  # Goal position
        print(maze)

    def is_game_over(self):
        return False  # No special game over condition for this simple example


def simulate(env):
    # Instantiating the learning-related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99

    num_streaks = 0

    # Render the maze
    env.render()

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
        total_reward = 0

        for t in range(MAX_T):

            # Select an action
            action = select_action(state_0, explore_rate, env)

            # Execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * best_q - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            if done:
                print(f"Episode {episode} finished after {t} time steps with total reward = {total_reward} (streak {num_streaks}).")

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print(f"Episode {episode} timed out at {t} with total reward = {total_reward}.")

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)


def select_action(state, explore_rate, env):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest Q
    else:
        action = int(np.argmax(q_table[state]))
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == "__main__":
    # Initialize the custom "maze" environment
    env = SimpleMazeEnv(size=(10, 10))

    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]

    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    # Learning related constants
    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

    # Simulation-related constants
    NUM_EPISODES = 5000
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    STREAK_TO_END = 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)

    # Creating a Q-Table for each state-action pair
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    # Begin simulation
    simulate(env)

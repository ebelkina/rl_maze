import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()

        self.grid_size = 10
        self.state = None
        self.sub_goal = (5, 5)  # Sub-goal at the center of the maze
        self.goal = (9, 9)  # Final goal at the bottom-right corner
        self.start = (0, 0)  # Starting point

        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)
        self.reset()

    def reset(self):
        self.state = self.start
        return np.array(self.state)

    def step(self, action):
        x, y = self.state

        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.grid_size - 1, y + 1)

        self.state = (x, y)

        reward = -1  # Default step penalty
        done = False

        # Check if agent reaches sub-goal or goal
        if self.state == self.sub_goal:
            reward = 10  # Reward for reaching the sub-goal
        if self.state == self.goal:
            reward = 100  # Reward for reaching the final goal
            done = True

        return np.array(self.state), reward, done, {}

    def render(self, mode="human"):
        maze = np.zeros((self.grid_size, self.grid_size))
        maze[self.state] = 2  # Mark agent's current position
        maze[self.sub_goal] = 1  # Mark sub-goal position
        maze[self.goal] = 3  # Mark final goal position

        # plt.imshow(maze, cmap='hot', interpolation='nearest')
        # plt.show()


import random

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state[0], state[1], action]
        next_max = np.max(self.q_table[next_state[0], next_state[1]])

        # Q-learning formula
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state[0], state[1], action] = new_value

    def train(self, episodes=1000):
        rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)

                state = next_state
                total_reward += reward

                self.env.render()

            rewards.append(total_reward)

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return rewards

if __name__ == "__main__":
    env = MazeEnv()
    agent = QLearningAgent(env)

    rewards = agent.train(episodes=500)

    # Plot the rewards over episodes
    plt.plot(rewards)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
    print("DONE")

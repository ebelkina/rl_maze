import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt


# Custom OpenAI Gym environment for the 10x10 maze
class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()
        self.maze = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        self.start = (1, 1)
        self.sub_goal = (5, 3)
        self.end_goal = (8, 7)

        self.action_space = spaces.Discrete(4)  # 4 possible directions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=9, shape=(2,), dtype=np.int32)

        self.state = self.start

    def reset(self):
        self.state = self.start
        return np.array(self.state)

    def step(self, action):
        # Map actions to changes in position
        actions_map = {
            0: (0, 1),  # right
            1: (0, -1),  # left
            2: (1, 0),  # down
            3: (-1, 0)  # up
        }

        # Compute next state
        next_state = (self.state[0] + actions_map[action][0], self.state[1] + actions_map[action][1])

        # Check if next position hits a wall (assuming walls are '1')
        if next_state == self.sub_goal:
            reward = 50  # Reaching the sub-goal
        elif next_state == self.end_goal:
            reward = 100  # Reaching the final goal
        elif self.maze[next_state] == 1:  # Wall
            reward = -100
            next_state = self.state  # Stay in place if hitting a wall
        else:
            reward = -1  # Default step penalty

        done = next_state == self.end_goal  # Episode ends when the agent reaches the final goal

        self.state = next_state
        return np.array(self.state), reward, done, {}

    def render(self):
        maze_copy = np.copy(self.maze)
        maze_copy[self.state] = 2  # Mark agent's current position
        maze_copy[self.end_goal] = 3  # Mark the final goal
        maze_copy[self.sub_goal] = 4  # Mark the sub-goal
        plt.imshow(maze_copy, cmap='hot', interpolation='nearest')
        plt.show()


# Q-learning algorithm
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = np.zeros((10, 10, env.action_space.n))  # Initialize Q-table for each state and action
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state[0], state[1]])  # Exploit

            next_state, reward, done, _ = env.step(action)
            best_future_q = np.max(q_table[next_state[0], next_state[1]])  # Best Q-value for next state
            q_table[state[0], state[1], action] += alpha * (
                        reward + gamma * best_future_q - q_table[state[0], state[1], action])
            state = next_state
        print(q_table)
        # Render the environment every 100 episodes to observe the learning progress
        if episode % 100 == 0:
            print(f"Episode: {episode}")
            env.render()

    return q_table


# Main function to run the maze environment and Q-learning
if __name__ == "__main__":
    env = MazeEnv()
    q_table = q_learning(env, episodes=2000)

    # Test the learned policy
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_table[state[0], state[1]])  # Follow the learned policy
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f"Total Reward: {total_reward}")

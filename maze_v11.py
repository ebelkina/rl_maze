import random

import gym
from gym import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
from gym.envs.registration import register
import time

class MazeEnv(gym.Env):
    def __init__(self, maze, alpha=0.1, gamma=0.99, epsilon=0.1, algorithm="random", experiment=0, show=True):
        super(MazeEnv, self).__init__()
        self.maze = np.array(maze)
        self.start_pos = (int(np.where(self.maze == 'S')[0]), int(np.where(self.maze == 'S')[1]))
        self.sub_goal_pos = (int(np.where(self.maze == 'G')[0]), int(np.where(self.maze == 'G')[1]))
        self.end_goal_pos = (int(np.where(self.maze == 'E')[0]), int(np.where(self.maze == 'E')[1]))
        self.current_pos = self.start_pos
        self.num_rows, self.num_cols = self.maze.shape

        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space is grid of size: rows x columns
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))

        # Initialize Pygame for visualization
        pygame.init()
        self.cell_size = 60
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size + 200, self.num_rows * self.cell_size))

        # Set font for displaying Q-values and button
        self.font = pygame.font.SysFont('Arial', 18)

        # Button properties
        self.button_color = (200, 200, 200)
        self.button_rect = pygame.Rect(self.num_cols * self.cell_size + 10, 10, 80, 40)
        self.is_paused = False

        # Q-learning and SARSA parameters
        self.algorithm = algorithm
        if self.algorithm == "q-learning" or self.algorithm == "sarsa":
            self.q_table = np.zeros((self.num_rows, self.num_cols, self.num_actions))  # For Q-learning and SARSA
        if self.algorithm == "policy_gradient":
            self.policy_table = np.ones((self.num_rows, self.num_cols, self.num_actions)) / self.num_actions  # Policy Gradient

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.reached_goals = set()

        self.episode = 0 # TODO
        self.total_reward = 0
        self.experiment = experiment
        self.show = show
        # self.seed = np.random.seed(experiment) # TODO
        self.done = False
        self.path = []

        self.arrow_up = pygame.image.load('arrow_up.png')
        self.arrow_down = pygame.image.load('arrow_down.png')
        self.arrow_left = pygame.image.load('arrow_left.png')
        self.arrow_right = pygame.image.load('arrow_right.png')

        # def load_arrow_images(self):
        #     """ Load arrow images for different directions. """
        #     self.arrow_up = pygame.image.load('arrow_up.png')
        #     self.arrow_down = pygame.image.load('arrow_down.png')
        #     self.arrow_left = pygame.image.load('arrow_left.png')
        #     self.arrow_right = pygame.image.load('arrow_right.png')
        #
        #     # Optionally, scale them to a base size
        #     base_size = (self.cell_size // 4, self.cell_size // 4)
        #     self.arrow_up = pygame.transform.scale(self.arrow_up, base_size)
        #     self.arrow_down = pygame.transform.scale(self.arrow_down, base_size)
        #     self.arrow_left = pygame.transform.scale(self.arrow_left, base_size)
        #     self.arrow_right = pygame.transform.scale(self.arrow_right, base_size)
        #
        # self.load_arrow_images()

    def reset(self, **kwargs):
        self.current_pos = self.start_pos
        return np.array(self.current_pos)

    def step(self, action):
        # Map actions to changes in position
        actions_map = {
            0: (0, 1),  # right
            1: (0, -1),  # left
            2: (1, 0),  # down
            3: (-1, 0)  # up
        }

        # Compute next state
        next_pos = (self.current_pos[0] + actions_map[action][0], self.current_pos[1] + actions_map[action][1])

        reward = -1 # Small penalty for regular movement to encourage efficiency
        self.done = False # TODO redudant?

        # Check if next state is within the maze bounds TODO not necessary
        if (0 <= next_pos[0] < self.num_rows) and (0 <= next_pos[1] < self.num_cols):
            # Check if the next state is a wall
            if self.maze[next_pos[0], next_pos[1]] == '1':
                reward = -100  # Penalty for hitting a wall
                next_pos = self.current_pos  # Stay in the same place if hitting a wall
            else:
                # Update the position
                self.current_pos = next_pos

                # Check if agent reaches the sub-goal
                if next_pos == self.sub_goal_pos and self.sub_goal_pos not in self.reached_goals:
                    reward = 5  # Reward for reaching the sub-goal TODO +50 only 1st time?
                    self.reached_goals.add(self.sub_goal_pos)  # Mark sub-goal as reached TODO just flag?

                # Check if agent reaches the final goal
                elif next_pos == self.end_goal_pos:
                    if self.sub_goal_pos in self.reached_goals:
                        reward = 100  # Reward for reaching the final goal
                    # else:
                    #     reward = 20
                    self.done = True # TODO add in reached goals?

        else:
            # If the agent tries to move out of bounds, stay in place and apply penalty TODO walls everythere >> no needed?
            reward = -100
            next_pos = self.current_pos  # Stay in the same place

        return np.array(self.current_pos), reward, self.done, None, {}

    def get_q_value_color(self, q_value):
        """ Map a Q-value to a color based on the current Q-table. """
        # Dynamically calculate the min and max Q-values from the Q-table
        q_min = np.min(self.q_table)  # Minimum Q-value in the entire Q-table
        q_max = np.max(self.q_table)  # Maximum Q-value in the entire Q-table

        # Normalize Q-value
        if q_max == q_min:
            normalized_q_value = 0.5  # Avoid division by zero, set it to a neutral value
        else:
            normalized_q_value = (q_value - q_min) / (q_max - q_min)

        # Map the normalized value to a color
        r = int(255 * (1 - normalized_q_value))  # Red decreases as Q-value increases
        g = int(255 * normalized_q_value)  # Green increases as Q-value increases
        return (r, g, 0)  # Return a color ranging from red to green

    def render(self):
        self.screen.fill((255, 255, 255))

        # Define the offset for positioning arrows within the cell
        arrow_offset = self.cell_size // 4  # Adjust this value if needed

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
                cell_center = (col * self.cell_size + self.cell_size // 2, row * self.cell_size + self.cell_size // 2)

                # Get the Q-values for the current cell
                q_values = self.q_table[row, col]

                # Define arrow images
                arrow_images = [self.arrow_right, self.arrow_left, self.arrow_down, self.arrow_up]

                # Draw arrows based on Q-values
                for i, q_value in enumerate(q_values):
                    # Calculate min and max Q-values
                    q_min = np.min(self.q_table)
                    q_max = np.max(self.q_table)

                    # Scale the size of the arrow based on Q-value
                    max_arrow_size = self.cell_size // 2
                    min_arrow_size = self.cell_size // 10

                    if q_max == q_min:
                        arrow_size = min_arrow_size  # Avoid division by zero by setting to a minimum size
                    else:
                        arrow_size = int(
                            min_arrow_size + (q_value - q_min) / (q_max - q_min) * (max_arrow_size - min_arrow_size))

                    # Scale the arrow image
                    scaled_arrow_image = pygame.transform.scale(arrow_images[i], (arrow_size, arrow_size))

                    # Position the arrow image
                    if i == 0:  # Right
                        self.screen.blit(scaled_arrow_image,
                                         (cell_center[0] + arrow_offset // 2, cell_center[1] - arrow_size // 2))
                    elif i == 1:  # Left
                        self.screen.blit(scaled_arrow_image,
                                         (cell_center[0] - arrow_offset, cell_center[1] - arrow_size // 2))
                    elif i == 2:  # Down
                        self.screen.blit(scaled_arrow_image,
                                         (cell_center[0] - arrow_size // 2, cell_center[1] + arrow_offset // 4))
                    elif i == 3:  # Up
                        self.screen.blit(scaled_arrow_image,
                                         (cell_center[0] - arrow_size // 2, cell_center[1] - arrow_offset))

                # Draw the agent's current position as a yellow circle
                if np.array_equal(np.array(self.current_pos), np.array([row, col])):
                    pygame.draw.circle(self.screen, 'yellow', cell_center, self.cell_size // 4)

        pygame.display.update()

    def apply_color_overlay(self, image, color):
        """ Apply a color overlay on the given image. """
        colored_image = image.copy()
        colored_image = image.copy()
        colored_image.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
        return colored_image

    def toggle_pause(self):
        """ Toggles the paused state of the game. """
        self.is_paused = not self.is_paused

    def handle_events(self):
        """ Handle Pygame events, such as mouse clicks on the button. """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.button_rect.collidepoint(event.pos):
                    self.toggle_pause()

    def choose_action(self, state, q_table=None):  # TODO not qtable?
        if self.algorithm == "random":
            return np.random.choice([0, 1, 2, 3])  # Random action

        elif self.algorithm == "policy_gradient":
            row, col = state
            action_probabilities = self.policy_table[row, col]
            # Ensure action probabilities are non-negative and normalized
            action_probabilities = np.clip(action_probabilities, 1e-10, 1)  # Avoid negative values TODO
            return np.random.choice(np.arange(self.num_actions), p=action_probabilities)


        elif self.algorithm == "q-learning" or self.algorithm == "sarsa" and q_table is not None:
            if np.random.uniform(0, 1) < self.epsilon:  # TODO not for q-learning?
                return self.action_space.sample()  # Explore
            else: # Exploit
                row, col = state
                max_value = np.max(q_table[row, col])  # Find the max Q-value
                # Get all actions that have the max Q-value
                max_actions = np.where(q_table[row, col] == max_value)[0]
                print(q_table[row, col])
                print('max', max_value)
                # Randomly choose one of the actions that have the max Q-value
                return np.random.choice(max_actions)

    def update_q_value_qlearning(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state
        best_next_action = np.argmax(self.q_table[next_row, next_col])
        # td_target = reward + self.gamma * self.q_table[next_row, next_col, best_next_action]
        # td_error = td_target - self.q_table[row, col, action]
        # self.q_table[row, col, action] += self.alpha * td_error
        self.q_table[row, col, action] += self.alpha * (reward +
                                                        self.gamma * self.q_table[next_row, next_col, best_next_action] -
                                                        self.q_table[row, col, action])

    def update_q_value_sarsa(self, state, action, reward, next_state, next_action):
        row, col = state
        next_row, next_col = next_state
        td_target = reward + self.gamma * self.q_table[next_row, next_col, next_action]
        td_error = td_target - self.q_table[row, col, action]
        self.q_table[row, col, action] += self.alpha * td_error

    def update_policy_gradient(self, state, action, reward):
        row, col = state
        gradient = np.zeros(self.num_actions)
        gradient[action] = reward
        self.policy_table[row, col] += self.alpha * gradient
        self.policy_table[row, col] /= np.sum(self.policy_table[row, col])  # Normalize to ensure valid probabilities

    def train(self, episodes=100, sleep_sec=0):
        random.seed(self.experiment)
        np.random.seed(self.experiment)
        total_rewards = []
        for episode in range(1, episodes+1):
            self.episode = episode
            state = self.reset()
            self.done = False
            self.total_reward = 0
            self.reached_goals = set()  # TODO

            q_table = None
            if self.algorithm == "q-learning":
                q_table = self.q_table
            elif self.algorithm == "sarsa":
                q_table = self.q_table

            self.path = [state]
            rewards_episode = []

            while not self.done:
                # Handle button events
                self.handle_events()
                # print('reached_goals', self.reached_goals)

                # Check if the game is paused
                if self.is_paused:
                    continue

                action = self.choose_action(state, q_table)

                next_state, reward, self.done, _, _ = self.step(action)

                if self.algorithm == "q-learning":
                    self.update_q_value_qlearning(state, action, reward, next_state)
                elif self.algorithm == "sarsa":
                    next_action = self.choose_action(next_state, self.q_table)
                    self.update_q_value_sarsa(state, action, reward, next_state, next_action)
                    action = next_action
                elif self.algorithm == "policy_gradient":
                    self.update_policy_gradient(state, action, reward) #, next_state)
                elif self.algorithm == "random":
                    pass  # No update for random walk

                # Render the environment to show training progress
                if self.show:
                    self.render()

                # Add a small delay for better visualization (adjust as needed)
                time.sleep(sleep_sec)

                # Move to the next state
                state = next_state
                self.total_reward += reward

                self.path.append(next_state)
                rewards_episode.append(reward)

                # # Render the environment to show training progress
                # self.render()
                #
                # # Add a small delay for better visualization (adjust as needed)
                # time.sleep(sleep_sec)

            print('total_reward', self.total_reward)
            print('path: ', self.path)
            print('rewards_episode: ', rewards_episode)
            print('reached_goals', self.reached_goals)

            # self.reached_goals = set()
            total_rewards.append(self.total_reward)
            # print(f"Episode {episode + 1}: Total Reward: {self.total_reward}")

        return total_rewards

    def visualize_learned_path(self):
        # Reset the environment to the start position
        state = self.reset()
        done = False

        # Loop until the agent reaches the goal
        while not done:
            # Handle button events
            self.handle_events()

            # Check if the game is paused
            if self.is_paused:
                continue

            # Get the best action based on Q-values
            row, col = state
            action = np.argmax(self.q_table[row, col])

            # Take the action and move the agent
            next_state, _, done, _, _ = self.step(action)

            # Render the environment to show the current position of the agent
            self.render()

            # Slow down the visualization to see the path
            time.sleep(0.5)  # Adjust the speed as necessary

            # Move to the next state
            state = next_state

        print("Reached the goal!")


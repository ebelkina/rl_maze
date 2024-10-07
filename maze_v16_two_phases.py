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
        # self.sub_goal_pos = (int(np.where(self.maze == 'G')[0]), int(np.where(self.maze == 'G')[1]))
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

        # self.reached_goals = set()

        self.episode = 0 # TODO
        self.total_reward = 0
        self.experiment = experiment
        self.show = show
        # self.seed = np.random.seed(experiment) # TODO
        self.done = False
        self.path = []
        self.image_counter = 1
        self.phase = 1

    def reset(self, change_phase=False):
        self.current_pos = self.start_pos
        if change_phase:
            self.q_table =  self.q_table = np.zeros((self.num_rows, self.num_cols, self.num_actions))
            self.phase = 2 # TODO
            self.image_counter = 1
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
                # next_pos = self.current_pos  # Stay in the same place if hitting a wall
                # next_pos = self.start_pos
                self.done = True
            else:
                # Update the position
                self.current_pos = next_pos

                # Check if agent reaches the sub-goal
                # if next_pos == self.sub_goal_pos and self.sub_goal_pos not in self.reached_goals:
                #     # reward = 50  # Reward for reaching the sub-goal TODO +50 only 1st time?
                #     self.reached_goals.add(self.sub_goal_pos)  # Mark sub-goal as reached TODO just flag?

                # # Check if agent reaches the final goal
                # elif next_pos == self.end_goal_pos:
                #     if self.sub_goal_pos in self.reached_goals:
                #         reward = 100  # Reward for reaching the final goal
                #     self.done = True # TODO add in reached goals?
                if next_pos == self.end_goal_pos:
                    reward = 100  # Reward for reaching the final goal
                    self.done = True # TODO add in reached goals?

        else:
            # If the agent tries to move out of bounds, stay in place and apply penalty TODO walls everythere >> no needed?
            reward = -100
            next_pos = self.current_pos  # Stay in the same place

        return np.array(self.current_pos), reward, self.done, None, {}

    def get_q_value_color(self, q_value):
        """ Map a Q-value to a color between white (low Q) and middle gray (high Q). """
        # Normalize Q-value to a range between 0 and 1
        normalized_q = (q_value - np.min(self.q_table)) / (np.max(self.q_table) - np.min(self.q_table) + 1e-5)
        # Interpolate between white (low) and middle gray (high)
        gray_value = int(255 - (normalized_q * (255 - 150)))  # The higher the Q-value, the closer to middle gray
        return (gray_value, gray_value, gray_value)

    def render(self):
        self.screen.fill((255, 255, 255))
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
                cell_center = (col * self.cell_size + self.cell_size // 2, row * self.cell_size + self.cell_size // 2)

                # Check for walls, start, sub-goal, and end goal
                if self.maze[row, col] == '1':
                    pygame.draw.rect(self.screen, 'black', (cell_left, cell_top, self.cell_size, self.cell_size))
                elif (row, col) == self.start_pos:
                    pygame.draw.rect(self.screen, 'green', (cell_left, cell_top, self.cell_size, self.cell_size))
                # elif (row, col) == self.sub_goal_pos:
                #     pygame.draw.rect(self.screen, 'blue', (cell_left, cell_top, self.cell_size, self.cell_size))
                elif (row, col) == self.end_goal_pos:
                    pygame.draw.rect(self.screen, 'red', (cell_left, cell_top, self.cell_size, self.cell_size))
                else:
                    # Get max Q-value for the current cell
                    max_q_value = np.max(self.q_table[row, col])
                    # Map Q-value to a color
                    cell_color = self.get_q_value_color(max_q_value)
                    pygame.draw.rect(self.screen, cell_color, (cell_left, cell_top, self.cell_size, self.cell_size))

                    # # Display the Q-value as text
                    # q_value_text = self.font.render(f'{max_q_value:.2f}', True, 'black')
                    # text_rect = q_value_text.get_rect(center=cell_center)
                    # self.screen.blit(q_value_text, text_rect)

                # Mark path if Done
                max_q_value = np.max(self.q_table[row, col])
                color_q_value = 'black'
                if any(np.array_equal((row, col), np.array(p)) for p in self.path):
                    color_q_value = 'red'
                q_value_text = self.font.render(f'{max_q_value:.2f}', True, color_q_value)
                text_rect = q_value_text.get_rect(center=cell_center)
                self.screen.blit(q_value_text, text_rect)

                ###
                # if self.done:
                #     for cell in self.path:
                #         if np.array_equal(cell, (row, col)):
                #             pygame.draw.circle(self.screen, 'red', cell_center, self.cell_size // 10)

                # Draw the agent's current position as a yellow circle
                if np.array_equal(np.array(self.current_pos), np.array([row, col])):
                    pygame.draw.circle(self.screen, 'yellow', cell_center, self.cell_size // 4)

        # Draw the pause button
        pygame.draw.rect(self.screen, self.button_color, self.button_rect)
        button_text = self.font.render('Pause' if not self.is_paused else 'Resume', True, 'black')
        self.screen.blit(button_text, button_text.get_rect(center=self.button_rect.center))

        # Calculate the position below the button to place the text
        text_x = self.button_rect.left  # Align text with left edge of the button
        text_y_start = self.button_rect.bottom + 10  # Start the text 10 pixels below the bottom of the button

        # Display information about train process
        algorithm_text = self.font.render(f'Algorithm: {self.algorithm}', True, (0, 0, 0))
        experiment_text = self.font.render(f'Experiment: {self.experiment}', True, (0, 0, 0))
        self.screen.blit(algorithm_text, (text_x + 10, text_y_start))
        self.screen.blit(experiment_text, (text_x + 10, text_y_start + 30))

        if self.show:
            phase = self.font.render(f'Phase: {self.phase}', True, (0, 0, 0))
            episode_text = self.font.render(f'Episode: {self.episode}', True, (0, 0, 0))
            reward_text = self.font.render(f'Total Reward: {self.total_reward}', True, (0, 0, 0))
            # reached_goals_text = self.font.render(f'Reached goals: {self.reached_goals}', True, (0, 0, 0))
            self.screen.blit(phase, (text_x + 10, text_y_start + 60))
            self.screen.blit(episode_text, (text_x + 10, text_y_start + 90))
            self.screen.blit(reward_text, (text_x + 10, text_y_start + 120))
            # self.screen.blit(reached_goals_text, (text_x + 10, text_y_start + 120))

        pygame.display.update()

        pygame.image.save(self.screen, f".\saved\ph{self.phase}_{self.image_counter}.png")
        self.image_counter += 1

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

        # elif self.algorithm == "policy_gradient":
        #     row, col = state
        #     action_probabilities = self.policy_table[row, col]
        #     # Ensure action probabilities are non-negative and normalized
        #     action_probabilities = np.clip(action_probabilities, 1e-10, 1)  # Avoid negative values TODO
        #     return np.random.choice(np.arange(self.num_actions), p=action_probabilities)


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

    def train(self, episodes=100, sleep_sec=0, start_pos=(1,1), goal=(8,7)):
        self.start_pos = start_pos
        self.end_goal_pos = goal
        random.seed(self.experiment)
        np.random.seed(self.experiment)
        total_rewards = []
        for episode in range(1, episodes+1):
            self.episode = episode
            state = self.reset()
            self.done = False
            self.total_reward = 0
            # self.reached_goals = set()  # TODO

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
            # print('reached_goals', self.reached_goals)

            # self.reached_goals = set()
            total_rewards.append(self.total_reward)
            # print(f"Episode {episode + 1}: Total Reward: {self.total_reward}")

        return total_rewards, self.q_table

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


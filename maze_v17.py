import random

import gym
from gym import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
from gym.envs.registration import register
import time

class MazeEnv(gym.Env):
    def __init__(self, maze, alpha=0.1, gamma=0.99, epsilon=0.1,
                 algorithm="random", experiment=0, show=True):
        super(MazeEnv, self).__init__()
        self.maze = np.array(maze)
        self.start_pos = (int(np.where(self.maze == 'S')[0]), int(np.where(self.maze == 'S')[1]))
        self.sub_goal_pos = (int(np.where(self.maze == 'G')[0]), int(np.where(self.maze == 'G')[1])) # TODO list of goals
        self.end_goal_pos = (int(np.where(self.maze == 'E')[0]), int(np.where(self.maze == 'E')[1]))
        self.current_pos = self.start_pos
        self.next_pos = self.current_pos
        self.num_rows, self.num_cols = self.maze.shape
        self.sub_goal_reached = False

        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space is grid of size: rows x columns
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))


        # Initialize Pygame for visualization
        pygame.init()
        self.cell_size = 60
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size + 400, self.num_rows * self.cell_size))

        # Set font for displaying Q-values and button
        self.font = pygame.font.SysFont('Arial', 18)
        self.small_font = pygame.font.SysFont('Arial', 10)

        # Button properties
        self.button_color = (200, 200, 200)
        self.button_rect = pygame.Rect(self.num_cols * self.cell_size + 10, 10, 80, 40)
        self.is_paused = False

        # Q-learning and SARSA parameters
        self.algorithm = algorithm
        if self.algorithm == "q-learning" or self.algorithm == "sarsa":
            self.q_table_1 = np.zeros((self.num_rows, self.num_cols, self.num_actions))  # For phase 1
            self.q_table_2 = np.zeros((self.num_rows, self.num_cols, self.num_actions))  # For phase 2
            # print('self.q_table[self.start_pos]\n', self.q_table[self.start_pos])
            self.q_table_current = self.get_q_table()
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
        self.path_1 = []
        self.path_2 = []
        self.current_path = self.path_1
        self.image_counter = 1
        # self.phase = 1
        self.max_num_steps_per_phase = 2 * self.maze.size # Assume TODO

    def get_q_table(self):
        if not self.sub_goal_reached:
            return self.q_table_1
        else:
            return self.q_table_2

    def reset(self, **kwargs):
        self.done = False
        self.total_reward = 0
        self.current_pos = self.start_pos
        self.q_table_current = self.q_table_1
        self.image_counter = 1
        self.sub_goal_reached = False
        self.q_table_current = self.q_table_1
        self.current_path = self.path_1 = []

        # if self.sub_goal_reached:
        #     self.current_pos = self.start_pos
        # if change_phase:
        #     self.phase = 2 # TODO
        #     self.image_counter = 1
        # return np.array(self.current_pos)

    def step(self, action):
        # Map actions to changes in position
        actions_map = {
            0: (-1, 0),  # up
            1: (0, 1),  # right
            2: (1, 0),  # down
            3: (0, -1),  # left
        }

        # Compute next state
        self.next_pos = (self.current_pos[0] + actions_map[action][0], self.current_pos[1] + actions_map[action][1])

        if self.show:
            print(f"self.next_pos, {self.next_pos} = ({self.current_pos[0]} + {actions_map[action][0]}, {self.current_pos[1]} + {actions_map[action][1]})")

        reward = -1 # Small penalty for regular movement to encourage efficiency
        self.done = False # TODO redudant?

        # Check if next state is within the maze bounds TODO not necessary
        if (0 <= self.next_pos[0] < self.num_rows) and (0 <= self.next_pos[1] < self.num_cols):
            # Check if the next state is a wall
            if self.maze[self.next_pos[0], self.next_pos[1]] == '1':
                reward = -100  # Penalty for hitting a wall
                # self.next_pos = self.current_pos  # Stay in the same place if hitting a wall
                # self.next_pos = self.start_pos
                # self.done = True # start from beginning
                if self.show:
                    print('wall >> state in same position')
            else:
            #     # Update the position
            #     self.current_pos = self.next_pos

                # Check if agent reaches the sub-goal
                if self.next_pos == self.sub_goal_pos and not self.sub_goal_reached: # not in self.reached_goals:
                    reward = 50  # Reward for reaching the sub-goal TODO +50 only 1st time?
                    # self.sub_goal_reached = True
                    # self.reached_goals.add(self.sub_goal_pos)  # Mark sub-goal as reached TODO just flag?

                # Check if agent reaches the final goal
                elif self.next_pos == self.end_goal_pos:
                    if self.sub_goal_reached:
                        reward = 100  # Reward for reaching the final goal
                    else:
                        reward = -10 #TODO
                    self.done = True # TODO add in reached goals?

        else:
            # If the agent tries to move out of bounds, stay in place and apply penalty TODO walls everythere >> no needed?
            reward = -100
            # self.next_pos = self.current_pos  # Stay in the same place
            if self.show:
                print('out of bounds >> state in same position')
        # return np.array(self.current_pos), reward, self.done, None, {}
        return np.array(self.next_pos), reward, self.done, None, {}

    def get_q_value_color(self, action):
        path = self.current_path
        if len(self.current_path) > 3:
            pass
        if action in self.current_path:
            return 'red'
        else:
            return 'black'

    def get_q_value_rect_color(self, q_value):
        """ Map a Q-value to a color between white (low Q) and middle gray (high Q). """
        # Normalize Q-value to a range between 0 and 1
        min = np.min(self.q_table_current)
        max = np.max(self.q_table_current)
        # first = (q_value - min) # TODO for debugging
        # second = (max - min + 1e-5)
        # result = first/second
        normalized_q = (q_value - min) / (np.max(self.q_table_current) - np.min(self.q_table_current) + 1e-5)
        # Interpolate between white (low) and middle gray (high)
        color_value = int(255 - (normalized_q * (255 - 100)))  # The higher the Q-value, the closer to middle gray
        if self.q_table_current is self.q_table_1:
            return (color_value, 255, color_value)
        elif self.q_table_current is self.q_table_2:
            return (color_value, color_value, 255)

    def choose_action(self):  # TODO not qtable?
        # if self.algorithm == "random":
        #     return np.random.choice([0, 1, 2, 3])  # Random action

        # elif self.algorithm == "policy_gradient":
        #     row, col = state
        #     action_probabilities = self.policy_table[row, col]
        #     # Ensure action probabilities are non-negative and normalized
        #     action_probabilities = np.clip(action_probabilities, 1e-10, 1)  # Avoid negative values TODO
        #     return np.random.choice(np.arange(self.num_actions), p=action_probabilities)

        if self.algorithm == "q-learning" or self.algorithm == "sarsa":
            if np.random.uniform(0, 1) < self.epsilon:  # TODO not for q-learning?
                return self.action_space.sample()  # Explore
            else: # Exploit
                row, col = self.current_pos
                max_value = np.max(self.q_table_current[row, col])  # Find the max Q-value
                # Get all actions that have the max Q-value
                max_actions = np.where(self.q_table_current[row, col] == max_value)[0]
                if self.show:
                    print('state:', row, col, "======================")
                    print('state:', row, col)
                    print(f'q_table_current[{row}, {col}]\n', self.q_table_current[row, col])
                    print('max', max_value)
                    print('max_actions', max_actions)
                # Randomly choose one of the actions that have the max Q-value
                chosen_action = np.random.choice(max_actions)
                if self.show:
                    print('chosen_action', chosen_action)
                return chosen_action
                # return np.argmax(self.q_table_current[row, col])

    def update_q_value_qlearning(self, action, reward, next_state):

        row, col = self.current_pos
        next_row, next_col = next_state
        best_next_action = np.argmax(self.q_table_current[next_row, next_col])
        # td_target = reward + self.gamma * self.q_table[next_row, next_col, best_next_action]
        # td_error = td_target - self.q_table[row, col, action]
        # self.q_table[row, col, action] += self.alpha * td_error
        diff = self.alpha * (reward + self.gamma * self.q_table_current[next_row, next_col, best_next_action] -
                             self.q_table_current[row, col, action])
        if self.show:
            print("next_state", next_state)
            print(f"q_table_current[{next_row}, {next_col}]\n", self.q_table_current[next_row, next_col])
            print("best_next_action", best_next_action)
            print(
                f"q_table[{row}, {col}, {action}] = {self.q_table_current[row, col, action]} + {self.alpha} * ({reward} + {self.gamma} * {self.q_table_current[next_row, next_col, best_next_action]} - {self.q_table_current[row, col, action]})",
                f"= {self.q_table_current[row, col, action] + diff}")

        self.q_table_current[row, col, action] += diff # TODO
        if self.show:
            print(f"updated_q-table for {row}, {col}\n", self.q_table_current[row, col])
            print(f"updated_q-table for {next_row}, {next_col}\n", self.q_table_current[next_row, next_col])



    def update_q_value_sarsa(self, action, reward, next_state, next_action):
        row, col = self.current_pos
        next_row, next_col = next_state
        td_target = reward + self.gamma * self.q_table_current[next_row, next_col, next_action]
        td_error = td_target - self.q_table_current[row, col, action]
        self.q_table_current[row, col, action] += self.alpha * td_error

    def update_policy_gradient(self, action, reward):
        row, col = self.current_pos
        gradient = np.zeros(self.num_actions)
        gradient[action] = reward
        self.policy_table[row, col] += self.alpha * gradient
        self.policy_table[row, col] /= np.sum(self.policy_table[row, col])  # Normalize to ensure valid probabilities

    def train(self, episodes=100, sleep_sec=0):
        random.seed(self.experiment)  # TODO doesn't work
        np.random.seed(self.experiment)

        total_rewards_in_episode = []

        for episode in range(1, episodes+1):
            self.episode = episode

            # total_rewards_phase_1 = []
            self.reset()

            # q_table = None
            # if self.algorithm == "q-learning":
            #     q_table = self.q_table
            # elif self.algorithm == "sarsa":
            #     q_table = self.q_table

            rewards_episode = []

            while not (self.done or len(self.current_path) > self.max_num_steps_per_phase):
                # Handle button events
                self.handle_events()

                # Check if the game is paused
                if self.is_paused:
                    continue

                if self.show:
                    self.render()

                # if self.algorithm == 'q-learning' or self.algorithm == 'sarsa': # TODO other alg?
                action = self.choose_action()

                next_state, reward_immidiate, self.done, _, _ = self.step(action)

                if self.algorithm == "q-learning":
                    self.update_q_value_qlearning(action, reward_immidiate, next_state)
                elif self.algorithm == "sarsa":
                    next_action = self.choose_action()
                    self.update_q_value_sarsa(action, reward_immidiate, self.next_pos, next_action)
                    action = next_action
                elif self.algorithm == "policy_gradient":
                    self.update_policy_gradient(action, reward_immidiate) #, next_state)

                if self.next_pos == self.sub_goal_pos and not self.sub_goal_reached: # not in self.reached_goals:
                    self.sub_goal_reached = True
                    self.q_table_current = self.q_table_2
                    self.current_path = self.path_2 = []

                self.current_path.append((self.current_pos, action))
                rewards_episode.append(reward_immidiate)

                if self.maze[self.next_pos[0], self.next_pos[1]] != '1': # TODO bounds, allowed
                    # Update the position
                    self.current_pos = self.next_pos

                # Render the environment to show training progress
                if self.show:
                    self.render()

                # Add a small delay for better visualization
                time.sleep(sleep_sec)



                # Move to the next state
                # self.current_pos = self.next_pos
                self.total_reward += reward_immidiate

            if self.show:
                print('total_reward', self.total_reward)
                print('path: ', self.current_path)
                print('rewards_episode: ', rewards_episode)
                # print('reached_goals', self.reached_goals)

            # self.reached_goals = set()
            total_rewards_in_episode.append(self.total_reward)
            # print(f"Episode {episode + 1}: Total Reward: {self.total_reward}")

        return total_rewards_in_episode, self.q_table_1, self.q_table_2

    def show_learned_path(self):
        # Reset the environment to the start position
        self.current_pos = self.start_pos
        self.q_table_current = self.q_table_1
        self.current_path = []
        self.done = False
        self.total_reward = 0
        self.sub_goal_reached = False
        self.epsilon = 0
        self.show = False

        # Loop until the agent reaches the goal
        while not self.done:
            # Handle button events
            self.handle_events()

            # Check if the game is paused
            if self.is_paused:
                continue

            # Get the best action based on Q-values
            row, col = self.current_pos
            action = np.argmax(self.q_table_current[row, col])
            print(self.q_table_current[row, col])
            print('action', action)

            # Take the action and move the agent
            next_state, reward_immidiate, self.done, _, _ = self.step(action)


            if self.next_pos == self.sub_goal_pos and not self.sub_goal_reached:  # not in self.reached_goals:
                self.sub_goal_reached = True
                self.q_table_current = self.q_table_2

            self.current_path.append(self.current_pos)
            print(self.current_path)

            if self.maze[self.next_pos[0], self.next_pos[1]] != '1':  # TODO bounds, allowed
                # Update the position
                self.current_pos = self.next_pos

            # Render the environment to show training progress
            self.render(show_learned_path=True)

            # Slow down the visualization to see the path
            time.sleep(0.05)

            # Move to the next state
            # self.current_pos = self.next_pos
            self.total_reward += reward_immidiate


        print("Reached the goal!")

    def render(self, show_learned_path=False):
        self.screen.fill((255, 255, 255))

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
                cell_center = (col * self.cell_size + 0.5 * self.cell_size,
                               row * self.cell_size + 0.5 * self.cell_size)

                # Check for walls, start, sub-goal, and end goal
                if self.maze[row, col] == '1':
                    pygame.draw.rect(self.screen, 'black', (cell_left, cell_top, self.cell_size, self.cell_size))
                else:
                    ### Display Q-table as 4 numbers in a cell
                    if (row, col) == self.start_pos:
                        pygame.draw.rect(self.screen, 'green', (cell_left, cell_top, self.cell_size, self.cell_size),
                                         10)
                    elif (row, col) == self.sub_goal_pos:
                        pygame.draw.rect(self.screen, 'blue', (cell_left, cell_top, self.cell_size, self.cell_size), 10)
                    elif (row, col) == self.end_goal_pos:
                        pygame.draw.rect(self.screen, 'red', (cell_left, cell_top, self.cell_size, self.cell_size), 10)
                    else:
                        pygame.draw.rect(self.screen, 'black', (cell_left, cell_top, self.cell_size, self.cell_size), 1)

                    if show_learned_path:
                        if (row, col) in self.current_path:
                            pygame.draw.circle(self.screen, 'red', cell_center, 0.1 * self.cell_size)
                    if self.show:
                        text_up = self.small_font.render(f'{self.q_table_current[row, col, 0]:.2f}', True,
                                                         self.get_q_value_color(((row, col), 0)))
                        text_right = self.small_font.render(f'{self.q_table_current[row, col, 1]:.2f}', True,
                                                            self.get_q_value_color(((row, col), 1)))
                        text_down = self.small_font.render(f'{self.q_table_current[row, col, 2]:.2f}', True,
                                                           self.get_q_value_color(((row, col), 2)))
                        text_left = self.small_font.render(f'{self.q_table_current[row, col, 3]:.2f}', True,
                                                           self.get_q_value_color(((row, col), 3)))

                        # Define positions for actions: up, right, down, left
                        cell_inside_up = (col * self.cell_size + 0.5 * self.cell_size,
                                          row * self.cell_size + 0.18 * self.cell_size)
                        cell_inside_right = (col * self.cell_size + 0.77 * self.cell_size,
                                             row * self.cell_size + 0.5 * self.cell_size)
                        cell_inside_down = (col * self.cell_size + 0.5 * self.cell_size,
                                            row * self.cell_size + 0.87 * self.cell_size)
                        cell_inside_left = (col * self.cell_size + 0.22 * self.cell_size,
                                            row * self.cell_size + 0.5 * self.cell_size)

                        # Draw gray rectangles behind the text for better visibility
                        rect_up = text_up.get_rect(center=cell_inside_up)
                        pygame.draw.rect(self.screen, self.get_q_value_rect_color(self.q_table_current[row, col, 0]), rect_up)
                        self.screen.blit(text_up, rect_up)

                        rect_right = text_right.get_rect(center=cell_inside_right)
                        pygame.draw.rect(self.screen, self.get_q_value_rect_color(self.q_table_current[row, col, 1]), rect_right)
                        self.screen.blit(text_right, rect_right)

                        rect_down = text_down.get_rect(center=cell_inside_down)
                        pygame.draw.rect(self.screen, self.get_q_value_rect_color(self.q_table_current[row, col, 2]), rect_down)
                        self.screen.blit(text_down, rect_down)

                        rect_left = text_left.get_rect(center=cell_inside_left)
                        pygame.draw.rect(self.screen, self.get_q_value_rect_color(self.q_table_current[row, col, 3]), rect_left)
                        self.screen.blit(text_left, rect_left)

                # # # Mark path if Done
                # for step in self.path_1:

                # max_q_value = np.max(self.q_table[row, col])
                # color_q_value = 'black'
                # if any(np.array_equal((row, col), np.array(p)) for p in self.path):
                #     color_q_value = 'red'
                # q_value_text = self.font.render(f'{max_q_value:.2f}', True, color_q_value)
                # text_rect = q_value_text.get_rect(center=cell_center)
                # self.screen.blit(q_value_text, text_rect)

                ###
                # if self.done:
                #     for cell in self.path:
                #         if np.array_equal(cell, (row, col)):
                #             pygame.draw.circle(self.screen, 'red', cell_center, self.cell_size // 10)

                # Draw the agent's current position as a red circle
                if np.array_equal(np.array(self.current_pos), np.array([row, col])):
                    pygame.draw.circle(self.screen, 'red', cell_center, 0.1 * self.cell_size)

        # Draw the pause button
        pygame.draw.rect(self.screen, self.button_color, self.button_rect)
        button_text = self.font.render('Pause' if not self.is_paused else 'Resume', True, 'black')
        self.screen.blit(button_text, button_text.get_rect(center=self.button_rect.center))

        # Calculate the position below the button to place the text
        text_x = self.button_rect.left  # Align text with left edge of the button
        text_y_start = self.button_rect.bottom + 10  # Start the text 10 pixels below the bottom of the button

        # Display information about train process
        # algorithm_text = self.font.render(f'Algorithm: {self.algorithm}_{self.epsilon}', True, (0, 0, 0))
        algorithm_text = self.font.render(f'Algorithm: {self.algorithm}', True, (0, 0, 0))
        experiment_text = self.font.render(f'Experiment: {self.experiment}', True, (0, 0, 0))
        self.screen.blit(algorithm_text, (text_x + 10, text_y_start))
        self.screen.blit(experiment_text, (text_x + 10, text_y_start + 30))

        if self.show:
            sub_goal_text = self.font.render(f'Sub goal reached', True, (0, 0, 0))
            episode_text = self.font.render(f'Episode: {self.episode}', True, (0, 0, 0))
            reward_text = self.font.render(f'Total Reward: {self.total_reward}', True, (0, 0, 0))


            if self.sub_goal_reached:
                self.screen.blit(sub_goal_text, (text_x + 10, text_y_start + 60))
            self.screen.blit(episode_text, (text_x + 10, text_y_start + 90))
            self.screen.blit(reward_text, (text_x + 10, text_y_start + 120))

        path_length_text = self.font.render(f'Path length: {len(self.current_path)}', True, (0, 0, 0))
        self.screen.blit(path_length_text, (text_x + 10, text_y_start + 150))

        pygame.display.update()

        pygame.image.save(self.screen, f".//saved//v17_{self.image_counter}.png")
        self.image_counter += 1

    def toggle_pause(self):
        """ Toggles the paused state of the game. """
        self.is_paused = not self.is_paused

    def handle_events(self):
        """ Handle Pygame events, such as mouse clicks on the button. """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # quit()  # TODO
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.button_rect.collidepoint(event.pos):
                    self.toggle_pause()


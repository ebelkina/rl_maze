import random
import gym
from gym import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
from gym.envs.registration import register
import time

class MazeEnv(gym.Env):
    def __init__(self, maze, alpha=0.1, gamma=0.99, epsilon=0.01,
                 algorithm="random", experiment=0, show=True, reduce_epsilon=False):
        super(MazeEnv, self).__init__()
        self.maze = np.array(maze)
        self.start_pos = (int(np.where(self.maze == 'S')[0]), int(np.where(self.maze == 'S')[1]))
        self.sub_goal_pos = (int(np.where(self.maze == 'G')[0]), int(np.where(self.maze == 'G')[1])) # TODO list of goals
        self.end_goal_pos = (int(np.where(self.maze == 'E')[0]), int(np.where(self.maze == 'E')[1]))
        self.current_state = self.start_pos
        self.next_state = self.current_state
        self.num_rows, self.num_cols = self.maze.shape
        self.sub_goal_reached = False

        # There are 4 possible actions: 0=up, 1=down, 2=left, 3=right
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
            self.q_table_1 = np.zeros((self.num_rows, self.num_cols, self.num_actions))
            self.q_table_2 = np.zeros((self.num_rows, self.num_cols, self.num_actions))
            self.q_table_current = self.get_q_table()
            self.counts_1 = np.ones((self.num_rows, self.num_cols, self.num_actions)) # 1 to avoid division by 0
            self.counts_2 = np.ones((self.num_rows, self.num_cols, self.num_actions))
            self.counts_current = self.counts_1

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.episode = 0
        self.total_reward = 0
        self.experiment = experiment
        self.show = show

        self.done = False
        self.path_1 = []
        self.path_2 = []
        self.current_path = self.path_1
        self.image_counter = 1
        self.max_num_steps_per_phase = 4 * self.maze.size # Assume TODO
        self.reduce_epsilon = reduce_epsilon

    def get_q_table(self):
        if not self.sub_goal_reached:
            return self.q_table_1
        else:
            return self.q_table_2

    def reset(self, **kwargs):
        self.done = False
        self.total_reward = 0
        self.current_state = self.start_pos
        self.q_table_current = self.q_table_1
        self.image_counter = 1
        self.sub_goal_reached = False
        self.q_table_current = self.q_table_1
        self.counts_current = self.counts_1
        self.current_path = self.path_1 = []

    def step(self, action, current_state):
        # Map actions to changes in position
        done = False

        actions_map = {
            0: (-1, 0),  # up
            1: (0, 1),  # right
            2: (1, 0),  # down
            3: (0, -1),  # left
        }

        # Compute next state
        self.next_state = (current_state[0] + actions_map[action][0], current_state[1] + actions_map[action][1])

        # if self.show:
        #     print(f"self.next_state, {self.next_state} = ({self.current_state[0]} + {actions_map[action][0]}, {self.current_state[1]} + {actions_map[action][1]})")

        reward = -1 # Small penalty for regular movement

        # Check if the next state is a wall
        if self.maze[self.next_state[0], self.next_state[1]] == '1':
            reward = -50  # Penalty for hitting a wall
            if self.show:
                print('wall >> state in same position')
        else:
            # Check if agent reaches the sub-goal for the first time
            if self.next_state == self.sub_goal_pos and not self.sub_goal_reached:
                reward = 20  # Small reward

            # Check if agent reaches the final goal
            elif self.next_state == self.end_goal_pos:
                if self.sub_goal_reached:
                    reward = 50  # Reward for reaching the final goal
                # else:
                #     reward = -10
                done = True

        # return np.array(self.next_state), reward, self.done, None, {}
        return self.next_state, reward, done, None, {}

    def get_q_value_color(self, action):
        return 'red' if action in self.current_path else 'black'

    def get_q_value_rect_color(self, q_value):
        """ Map a Q-value to a color between white (low Q) and middle green/blue (high Q).
        Greenish colors for Q-values from Start to Sub-Goal,
        Bluish colors for Q-values from Sub-Goal to End-Goal"""
        # Normalize Q-value to a range between 0 and 1
        min = np.min(self.q_table_current)
        max = np.max(self.q_table_current)
        normalized_q = (q_value - min) / (max - min + 1e-5)
        # Interpolate between white (low) and middle green/blue (high)
        color_value = int(255 - (normalized_q * (255 - 100)))  # The higher the Q-value, the closer to middle green/blue

        if self.q_table_current is self.q_table_1:
            return (color_value, 255, color_value)
        elif self.q_table_current is self.q_table_2:
            return (color_value, color_value, 255)

    def choose_action(self, from_state):

        if self.algorithm == "q-learning" or self.algorithm == "sarsa":
            if np.random.uniform(0, 1) < self.epsilon: # Explore
                # chosen_action = self.action_space.sample()
                exploration_rate = 1/self.counts_current[self.current_state]
                chosen_action = np.random.choice(len(self.q_table_current[self.current_state]),
                                                 p=exploration_rate/np.sum(exploration_rate))

            else: # Exploit
                row, col = from_state
                # Find the max Q-value for current state
                max_value = np.max(self.q_table_current[row, col])
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


    def update_q_value_qlearning(self, action, reward, next_state):

        row, col = self.current_state
        next_row, next_col = next_state
        best_next_action = np.argmax(self.q_table_current[next_row, next_col])
        difference = self.alpha * (reward +
                                   self.gamma * self.q_table_current[next_row, next_col, best_next_action] -
                                   self.q_table_current[row, col, action])
        if self.show:
            print("next_state", next_state)
            print(f"q_table_current[{next_row}, {next_col}]\n", self.q_table_current[next_row, next_col])
            print("best_next_action", best_next_action)
            print(
                f"q_table[{row}, {col}, {action}] = {self.q_table_current[row, col, action]} + {self.alpha} * ({reward} + {self.gamma} * {self.q_table_current[next_row, next_col, best_next_action]} - {self.q_table_current[row, col, action]})",
                f"= {self.q_table_current[row, col, action] + difference}")

        self.q_table_current[row, col, action] += difference
        self.counts_current[row, col, action] += 1

        if self.show:
            print(f"updated_q-table for {row}, {col}\n", self.q_table_current[row, col])
            print(f"updated_q-table for {next_row}, {next_col}\n", self.q_table_current[next_row, next_col])

    def update_q_value_sarsa(self, action, reward, next_state, next_action):
        next_state, reward_immediate, self.done, _, _ = self.step(action, self.current_state)
        row, col = self.current_state
        next_row, next_col = next_state
        # td_target = reward + self.gamma * self.q_table_current[next_row, next_col, next_action]
        # td_error = td_target - self.q_table_current[row, col, action]
        # self.q_table_current[row, col, action] += self.alpha * td_error

        difference = self.alpha * (reward +
                                   self.gamma * self.q_table_current[next_row, next_col, next_action] -
                                   self.q_table_current[row, col, action])

        if self.show:
            print("next_state", next_state)
            print(f"q_table_current[{next_row}, {next_col}]\n", self.q_table_current[next_row, next_col])
            print("next_action", next_action)
            print(
                f"q_table[{row}, {col}, {action}] = {self.q_table_current[row, col, action]} + {self.alpha} * ({reward} + {self.gamma} * {self.q_table_current[next_row, next_col, next_action]} - {self.q_table_current[row, col, action]})",
                f"= {self.q_table_current[row, col, action] + difference}")

        self.q_table_current[row, col, action] += difference
        self.counts_current[row, col, action] += 1

        if self.show:
            print(f"updated_q-table for {row}, {col}\n", self.q_table_current[row, col])
            print(f"updated_q-table for {next_row}, {next_col}\n", self.q_table_current[next_row, next_col])


    def train(self, episodes=100, sleep_sec=0, opt_path_reward=0, opt_path_len=0):
        random.seed(self.experiment)
        np.random.seed(self.experiment)

        total_rewards_in_episodes = []
        path_length_in_episodes = []
        optimal_path_found = []

        for episode in range(1, episodes+1):
            self.episode = episode


            self.reset()
            rewards_episode = []

            # For SARSA: Choose action A from S using policy derived from Q (e-greedy/count-based) TODO
            # if self.algorithm == "sarsa":
            #     action = self.choose_action(from_state=self.current_state)

            # Loop until the agent reaches the Sub-Goal and End-Goal or path is too long for this phase
            while not (self.done or len(self.current_path) > self.max_num_steps_per_phase):
                self.handle_events() # Handle button events
                if self.is_paused: # Check if the game is paused
                    continue
                if self.show:
                    self.render()

                # For Q-learning: Choose action A from S using policy derived from Q (e-greedy/count-based) TODO
                # if self.algorithm == "q-learning":
                action = self.choose_action(from_state=self.current_state) # based on e-greedy/count-based

                # Take action A, observe R and S'
                next_state, reward_immediate, self.done, _, _ = self.step(action, self.current_state)

                # Update Q-value
                if self.algorithm == "q-learning": # based on max value (so epsilon=0)
                    self.update_q_value_qlearning(action, reward_immediate, next_state)
                elif self.algorithm == "sarsa": # based on e-greedy/count-based
                    next_action = self.choose_action(from_state=next_state)  # update action for the next step for SARSA based on e-greedy/count-based
                    self.update_q_value_sarsa(action, reward_immediate, next_state, next_action)
                    # action = next_action # TODO ???? update action for the next step for SARSA

                if self.next_state == self.sub_goal_pos and not self.sub_goal_reached:
                    self.sub_goal_reached = True
                    self.q_table_current = self.q_table_2
                    self.counts_current = self.counts_2
                    self.current_path = self.path_2 = []

                self.current_path.append((self.current_state, action))
                rewards_episode.append(reward_immediate)

                # Move to the next state if it's not a wall
                if self.maze[self.next_state[0], self.next_state[1]] != '1':
                # if self.maze[self.next_state] != '1':
                    self.current_state = self.next_state

                self.total_reward += reward_immediate

                if self.show:
                    self.render()
                time.sleep(sleep_sec) # Add a small delay for better visualization

            if self.show:
                print('total_reward', self.total_reward)
                print('path: ', self.current_path)
                print('rewards_episode: ', rewards_episode)
                # print('reached_goals', self.reached_goals)

            if self.reduce_epsilon:
                self.epsilon *= 0.95

            learned_path, learned_path_reward = self.check_learned_path(opt_path_reward, opt_path_len)
            if learned_path_reward == opt_path_reward:
                # print("learned_path LEN", len(learned_path))
                # print("learned_path_reward", learned_path_reward)
                # print('episode', episode)
                optimal_path_found.append(1)
            else:
                optimal_path_found.append(0)

            total_rewards_in_episodes.append(self.total_reward)
            path_length_in_episodes.append(len(self.current_path))


        return total_rewards_in_episodes, path_length_in_episodes, optimal_path_found, self.q_table_1, self.q_table_2

    def check_learned_path(self,opt_path_reward, opt_path_len):
        # Reset the environment but keep needed parameters (so it's not the standard reset)
        current_state = self.start_pos
        q_table = self.q_table_1  # copy?
        learned_path = []
        done = False
        total_reward = 0
        sub_goal_reached = False
        epsilon = 0
        show = True#False

        # Loop until the agent reaches the Sub-Goal and End-Goal or whole path is too long
        while not done and len(learned_path) <= opt_path_len:
            self.handle_events() # Handle button events
            if self.is_paused: # Check if the game is paused
                continue

            # Take the best action based on Q-values (greedy) to show the best path found
            row, col = current_state
            action = np.argmax(q_table[row, col])
            next_state, reward_immediate, done, _, _ = self.step(action, current_state)
            # print('current_state', current_state)
            # print('q_table[row, col]', q_table[row, col])
            # print('action', action)
            # print('next_state', next_state)
            # print('reward_immediate', reward_immediate)

            # Switch to q_table_2 if Sub-Goal is reached
            if next_state == self.sub_goal_pos and not sub_goal_reached:
                sub_goal_reached = True
                q_table = self.q_table_2

            learned_path.append(current_state)
            total_reward += reward_immediate
            # print('learned_path', learned_path)
            # print('total_reward', total_reward)

            # Move to the next state if it's not a wall
            if self.maze[next_state] != '1':
                current_state = next_state
                # print('OLD next_state', next_state)
                # print('current_state', current_state)

            # self.render(show_learned_path=True)
            # time.sleep(0.05) # Slow down the visualization to see the path
        return learned_path, total_reward

    def render(self, show_learned_path=False):
        self.screen.fill((255, 255, 255))

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
                cell_center = (col * self.cell_size + 0.5 * self.cell_size,
                               row * self.cell_size + 0.5 * self.cell_size)

                # Draw cells
                if self.maze[row, col] == '1': # Walls: totally black
                    pygame.draw.rect(self.screen, 'black', (cell_left, cell_top, self.cell_size, self.cell_size))
                else:
                    if (row, col) == self.start_pos: # Start: green
                        pygame.draw.rect(self.screen, 'green',(cell_left, cell_top, self.cell_size, self.cell_size), 10)
                    elif (row, col) == self.sub_goal_pos: # Sub-Goal: blue
                        pygame.draw.rect(self.screen, 'blue', (cell_left, cell_top, self.cell_size, self.cell_size), 10)
                    elif (row, col) == self.end_goal_pos: # End Goal: red
                        pygame.draw.rect(self.screen, 'red', (cell_left, cell_top, self.cell_size, self.cell_size), 10)
                    else: # Other cells: white with black with outline
                        pygame.draw.rect(self.screen, 'black', (cell_left, cell_top, self.cell_size, self.cell_size), 1)

                    # Draw learned path as red circles
                    if show_learned_path:
                        if (row, col) in self.current_path:
                            pygame.draw.circle(self.screen, 'red', cell_center, 0.1 * self.cell_size)

                    if self.show:
                        ### Display Q-table as 4 numbers in a cell (up, right, down, left)
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

                        # Draw rectangles behind the text with color intensity depending on value
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

                # Draw the agent's current position as a red circle
                if np.array_equal(np.array(self.current_state), np.array([row, col])):
                    pygame.draw.circle(self.screen, 'red', cell_center, 0.1 * self.cell_size)

        # Draw the pause button
        pygame.draw.rect(self.screen, self.button_color, self.button_rect)
        button_text = self.font.render('Pause' if not self.is_paused else 'Resume', True, 'black')
        self.screen.blit(button_text, button_text.get_rect(center=self.button_rect.center))
        text_x = self.button_rect.left  # Align text with left edge of the button
        text_y_start = self.button_rect.bottom + 10  # Start the text 10 pixels below the bottom of the button

        # Display information about train process
        algorithm_text = self.font.render(f'Algorithm: {self.algorithm}', True, (0, 0, 0))
        experiment_text = self.font.render(f'Experiment: {self.experiment}', True, (0, 0, 0))
        self.screen.blit(algorithm_text, (text_x + 10, text_y_start))
        self.screen.blit(experiment_text, (text_x + 10, text_y_start + 30))

        if self.show:
            sub_goal_text = self.font.render(f'Sub goal reached', True, (0, 0, 0))
            episode_text = self.font.render(f'Episode: {self.episode}', True, (0, 0, 0))
            reward_text = self.font.render(f'Total Reward: {self.total_reward}', True, (0, 0, 0))
            if self.sub_goal_reached:  # Indicate if Sub-Goal is reached
                self.screen.blit(sub_goal_text, (text_x + 10, text_y_start + 60))
            self.screen.blit(episode_text, (text_x + 10, text_y_start + 90))
            self.screen.blit(reward_text, (text_x + 10, text_y_start + 120))
        path_length_text = self.font.render(f'Path length: {len(self.current_path)}', True, (0, 0, 0))
        self.screen.blit(path_length_text, (text_x + 10, text_y_start + 150))

        pygame.display.update()

        # Save each rendering as png for algorithm verification
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


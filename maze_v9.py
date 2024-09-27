import gym
from gym import spaces
import numpy as np
import pygame
from gym.envs.registration import register
import time

class MazeEnv(gym.Env):
    def __init__(self, maze, alpha=0.1, gamma=0.99, epsilon=0.1):
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
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size + 100, self.num_rows * self.cell_size))

        # Set font for displaying Q-values and button
        self.font = pygame.font.SysFont('Arial', 18)

        # Button properties
        self.button_color = (200, 200, 200)
        self.button_rect = pygame.Rect(self.num_cols * self.cell_size + 10, 10, 80, 40)
        self.is_paused = False

        # Q-learning parameters
        self.q_table = np.zeros((self.num_rows, self.num_cols, self.num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.reached_goals = set()

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

        reward = 0
        done = False

        # Check if next state is within the maze bounds
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
                    reward = -1 # 50  # Reward for reaching the sub-goal
                    self.reached_goals.add(self.sub_goal_pos)  # Mark sub-goal as reached

                # Check if agent reaches the final goal
                elif next_pos == self.end_goal_pos and self.sub_goal_pos in self.reached_goals:
                    reward = 100  # Reward for reaching the final goal
                    done = True

                else:
                    reward = -1  # Small penalty for regular movement to encourage efficiency

        else:
            # If the agent tries to move out of bounds, stay in place and apply penalty
            reward = -100
            next_pos = self.current_pos  # Stay in the same place

        return np.array(self.current_pos), reward, done, None, {}

    def get_q_value_color(self, q_value):
        """ Map a Q-value to a color between red (low Q) and green (high Q). """
        # Normalize Q-value to a range between 0 and 1
        normalized_q = (q_value - np.min(self.q_table)) / (np.max(self.q_table) - np.min(self.q_table) + 1e-5)
        # Interpolate between red (low) and green (high)
        red = int((1 - normalized_q) * 255)
        green = int(normalized_q * 255)
        return (red, green, 0)

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
                elif (row, col) == self.sub_goal_pos:
                    pygame.draw.rect(self.screen, 'blue', (cell_left, cell_top, self.cell_size, self.cell_size))
                elif (row, col) == self.end_goal_pos:
                    pygame.draw.rect(self.screen, 'red', (cell_left, cell_top, self.cell_size, self.cell_size))
                else:
                    # Get max Q-value for the current cell
                    max_q_value = np.max(self.q_table[row, col])
                    # Map Q-value to a color
                    cell_color = self.get_q_value_color(max_q_value)
                    pygame.draw.rect(self.screen, cell_color, (cell_left, cell_top, self.cell_size, self.cell_size))

                    # Display the Q-value as text
                    q_value_text = self.font.render(f'{max_q_value:.2f}', True, (0, 0, 0))
                    text_rect = q_value_text.get_rect(center=cell_center)
                    self.screen.blit(q_value_text, text_rect)

                # Draw the agent's current position as a yellow circle
                if np.array_equal(np.array(self.current_pos), np.array([row, col])):
                    pygame.draw.circle(self.screen, 'yellow', cell_center, self.cell_size // 4)

        # Draw the pause button
        pygame.draw.rect(self.screen, self.button_color, self.button_rect)
        button_text = self.font.render('Pause' if not self.is_paused else 'Resume', True, (0, 0, 0))
        self.screen.blit(button_text, button_text.get_rect(center=self.button_rect.center))

        pygame.display.update()

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

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return self.action_space.sample()
        else:
            # Exploit: choose the action with max Q-value
            row, col = state
            return np.argmax(self.q_table[row, col])

    def update_q_value(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state
        best_next_action = np.argmax(self.q_table[next_row, next_col])
        td_target = reward + self.gamma * self.q_table[next_row, next_col, best_next_action]
        td_error = td_target - self.q_table[row, col, action]
        self.q_table[row, col, action] += self.alpha * td_error

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.reset()
            done = False
            total_reward = 0

            while not done:
                # Handle button events
                self.handle_events()

                # Check if the game is paused
                if self.is_paused:
                    continue

                # Choose action
                action = self.choose_action(state)

                # Take a step
                next_state, reward, done, _, _ = self.step(action)

                # Update Q-values
                self.update_q_value(state, action, reward, next_state)

                # Render the environment to show training progress
                self.render()

                # Add a small delay for better visualization (adjust as needed)
                time.sleep(0.1)

                # Move to the next state
                state = next_state
                total_reward += reward

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

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

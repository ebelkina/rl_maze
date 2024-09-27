import gym
from gym import spaces
import numpy as np
import pygame


class MazeGameEnv(gym.Env):
    def __init__(self, maze):
        super(MazeGameEnv, self).__init__()
        self.maze = np.array(maze)  # Maze represented as a 2D numpy array
        # self.start_pos = np.where(self.maze == 'S')  # Starting position
        # self.goal_pos = np.where(self.maze == 'G')  # Goal position

        self.start_pos = (int(np.where(self.maze == 'S')[0]), int(np.where(self.maze == 'S')[1]))  # Starting position as (row, col)
        self.goal_pos = (
        int(np.where(self.maze == 'G')[0]), int(np.where(self.maze == 'G')[1]))  # Goal position as (row, col)
        # self.end_goal_pos = (int(np.where(self.maze == 'E')[0]), int(np.where(self.maze == 'E')[1]))  # Goal position as (row, col)

        self.current_pos = self.start_pos  # starting position is current positon of agent
        self.num_rows, self.num_cols = self.maze.shape

        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation space is grid of size:rows x columns
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))

        # Initialize Pygame
        pygame.init()
        self.cell_size = 60

        # setting display size
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def reset(self, **kwargs):
        self.current_pos = self.start_pos
        return np.array(self.current_pos)

    def step(self, action):
        # Move the agent based on the selected action
        new_pos = np.array(self.current_pos)
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos

        # Reward function
        if np.array_equal(self.current_pos, self.goal_pos):
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        return np.array(self.current_pos), reward, done, {}

    def _is_valid_position(self, pos):
        row, col = pos

        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        # If the agent hits an obstacle
        if self.maze[row, col] == '1':
            return False
        return True

    def render(self):
        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
                cell_center = (col * self.cell_size + self.cell_size // 2, row * self.cell_size + self.cell_size // 2)

                try:
                    print(np.array(self.current_pos) == np.array([row, col]))  #.reshape(-1, 1))
                except Exception as e:
                    print('Initial state')

                if self.maze[row, col] == '1':  # Obstacle
                    pygame.draw.rect(self.screen, 'black', (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'S':  # Starting position
                    pygame.draw.rect(self.screen, 'green', (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'G':  # Sub-goal position
                    pygame.draw.rect(self.screen, 'blue', (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'E':  # End Goal position, the agent should reach before continuing
                    pygame.draw.rect(self.screen, 'red', (cell_left, cell_top, self.cell_size, self.cell_size))

                if np.array_equal(np.array(self.current_pos), np.array([row, col])):  # Agent position
                    pygame.draw.circle(self.screen, 'yellow', cell_center, self.cell_size // 4)

        pygame.display.update()  # Update the display

    # def get_valid_actions(self):
    #     possible_actions = list(range(self.action_space.n))  # Get all possible actions from env.action_space
    #     valid_actions = []
    #
    #     # Get the agent's current position
    #     current_pos = np.array(env.current_pos)
    #
    #     # Check each action and see if it leads to a valid position
    #     for action in possible_actions:
    #         new_pos = np.array(current_pos)  # Start from the current position
    #
    #         if action == 0:  # Up
    #             new_pos[0] -= 1
    #         elif action == 1:  # Down
    #             new_pos[0] += 1
    #         elif action == 2:  # Left
    #             new_pos[1] -= 1
    #         elif action == 3:  # Right
    #             new_pos[1] += 1
    #
    #         # Check if the new position is valid
    #         if env._is_valid_position(new_pos):
    #             valid_actions.append(action)  # Add to the list of valid actions
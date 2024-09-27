# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb

from IPython.display import IFrame
import numpy as np
import gym
import random

# Displaying the YouTube video using IFrame
IFrame('https://www.youtube.com/embed/q2ZOEFAaaI0?showinfo=0', width=560, height=315)

# Gym environment setup
env = gym.make("FrozenLake-v1", is_slippery=True)

action_size = env.action_space.n
state_size = env.observation_space.n

# Initializing Q-table
qtable = np.zeros((state_size, action_size))

# Display the Q-table
print(qtable)

############# Hyperparameters

total_episodes = 15000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005            # Exponential decay rate for exploration prob

############# Q-Learning algorithm
# List of rewards
rewards = []

# Q-learning algorithm
for episode in range(total_episodes):
    # Reset the environment and get the initial state
    state, _ = env.reset()
    step = 0
    total_rewards = 0

    for step in range(max_steps):
        # Choose an action a in the current world state (s)
        exp_exp_tradeoff = random.uniform(0, 1)

        # Exploitation: take action with the maximum Q value (greedy)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        # Exploration: take random action
        else:
            action = env.action_space.sample()

        # Take the action and observe the new state and reward
        new_state, reward, terminated, truncated, info = env.step(action)

        # Update Q(s,a) using the Q-learning formula
        qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards += reward

        # Move to the next state
        state = new_state

        # If the episode is terminated or truncated (i.e., the agent is done), break the loop
        if terminated or truncated:
            break

    # Reduce epsilon as we want to decrease exploration over time
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

# Print the average reward over all episodes
print("Score over time: " + str(sum(rewards) / total_episodes))
print("Q-Table:")
print(qtable)

######### Use the learned Q-table to play FrozenLake

for episode in range(5):
    state, _ = env.reset()
    step = 0
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        # Choose the action with the highest Q value
        action = np.argmax(qtable[state, :])

        # Take the action and move to the new state
        new_state, reward, terminated, truncated, info = env.step(action)

        # Render the environment to visualize the agent
        env.render()

        # If done, either by reaching the goal or falling into a hole, stop the episode
        if terminated or truncated:
            print("Number of steps:", step)
            break

        # Move to the new state
        state = new_state

env.close()

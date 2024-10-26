import gym
import pygame
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from maze_v19 import MazeEnv
from scipy.stats import ttest_rel, ttest_ind
import copy
import wandb
import os

# Register the custom environment
gym.register(
    id='Maze_v19',
    entry_point='maze_v19:MazeEnv',
    kwargs={'maze': None}
)

# Define the maze: 1 - wall, 'S' - start, 'E' - end-goal, 0 - empty space
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'S', 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 'E', 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Define possible sub-goal positions and data for testing calculated manually
# TODO check values
sub_goals_map = {
    (5, 3): {'opt_path_reward': 36, 'opt_path_len': 15},
    # (7, 3): {'opt_path_reward': 32, 'opt_path_len': 19},
    (1, 4): {'opt_path_reward': 32, 'opt_path_len': 19},
    (8, 3): {'opt_path_reward': 30, 'opt_path_len': 21},
    # (8, 2): {'opt_path_reward': 28, 'opt_path_len': 23},
    # (1, 7): {'opt_path_reward': 28, 'opt_path_len': 23},
}
sub_goals = list(sub_goals_map.keys())


### Parameters
id = '01'  # experiment id
num_runs = 100  # equal to random seed for repeatability TODO check if it works
episodes = 200
algorithms = ["q-learning", "sarsa"]
epsilons = [0.1]
reduce_epsilon = True
alpha = 0.05
gamma = 0.99

# show_training = False
show_training = True # commit png out if want to automatically save pngs for one run
sleep_sec = 0
# sleep_sec = 0.05  # slow down simulation
show_learned_path = True # TODO fix it (it used to work before I added checking path after each episode)
save_raw_data = False

# Initialize a dictionary to store the results for each algorithm
# e.g. keys: {((5, 3), 'q-learning', 0.1): [], ((5, 3), 'sarsa', 0.1): [], ((7, 3), 'q-learning', 0.1): [], ...
results = {(sub_goal, alg, eps): [] for sub_goal in sub_goals_map.keys() for alg in algorithms for eps in epsilons}
convergence_episodes = {(sub_goal, alg): [] for sub_goal in sub_goals_map.keys() for alg in algorithms}

folder = "./results"
if not os.path.exists(folder):
    os.makedirs(folder)

### Run experiment

# Set a sub-goal in the maze
def place_sub_goal(maze, sub_goal_pos):
    row, col = sub_goal_pos
    new_maze = copy.deepcopy(maze)
    # Check if sub-goal position is empty (not wall, start or end-goal)
    if new_maze[row][col] == 0:
        new_maze[row][col] = 'G'
        return new_maze
    else:
        print("Sub-goal cannot be places")

for sub_goal, sub_goal_data in sub_goals_map.items():
    opt_path_reward = sub_goal_data['opt_path_reward']
    opt_path_len = sub_goal_data['opt_path_len']
    print(f'\nRunning for Sub-Goal {sub_goal} - Optimal path reward: {opt_path_reward}, Optimal path length: {opt_path_len}')

    maze_with_sub_goal = place_sub_goal(maze, sub_goal)

    output_folder = f'{folder}/{sub_goal}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for alg in algorithms:
        for eps in epsilons:
            convergence_episode = 1000  # if cannot convergence within 200 episodes TODO improve
            for run in range(1, num_runs + 1):

                # #######################
                # # Initialize W&B for this run
                # wandb.init(
                #     project="maze-rl-2",
                #     name=f"{id}_{alg}_{sub_goal}_run{run}",
                #     config={
                #         "algorithm": alg,
                #         "epsilon": eps,
                #         "alpha": alpha,
                #         "episodes": episodes,
                #         "sub_goal": sub_goal,
                #         "opt_path_reward": opt_path_reward,
                #         "opt_path_len": opt_path_len,
                #         "run_number": run
                #     }
                # )
                # ##########################

                # Create environment
                env = gym.make('Maze_v19', maze=maze_with_sub_goal, epsilon=eps, algorithm=alg, run=run, show_training=show_training,
                               show_learned_path=show_learned_path,
                               reduce_epsilon=reduce_epsilon, alpha=alpha,
                               gamma=gamma, output_folder=output_folder)
                env.reset()
                env.render()

                # Train and log data
                rewards, path_lengths, optimal_path_found, q_table_1, q_table_2 = env.train(episodes,
                                                                                            sleep_sec,
                                                                                            opt_path_reward,
                                                                                            opt_path_len
                                                                                            )
                # ################ Log data to W&B after each episode
                # for episode in range(episodes):
                #     chart_name=f"{id}_{alg}_{sub_goal}"
                #
                #     wandb.log({
                #         f'Episode {chart_name}': episode + 1,
                #         f'Reward {chart_name}': rewards[episode],
                #         f'Path_Length {chart_name}': path_lengths[episode],
                #         f'Optimal_Path_Found {chart_name}': optimal_path_found[episode]
                #     })
                #

                #
                # # Save artifacts (e.g., model, Q-tables) if necessary
                # # You can log other files like models or configurations as needed
                # wandb.save(csv_filename)
                #
                # # Finish the W&B run
                # wandb.finish()
                # #####################

                # Determine the episode of convergence for each run
                for episode in range(10, episodes):
                    # Check if `optimal_path_found` was true 8 out of the last 10 episodes
                    if np.sum(optimal_path_found[episode - 10:episode]) >= 8:
                        convergence_episode = episode
                        break
                else:
                    convergence_episode = 1000  # TODO 1000
                    print('convergence_episode = 1000')

                convergence_episodes[(sub_goal, alg)].append(convergence_episode)
                results[(sub_goal, alg, eps)].append((rewards, path_lengths, optimal_path_found))

                df_results = pd.DataFrame({
                    'Episode': range(1, len(rewards) + 1),
                    'Reward': rewards,
                    'Path_Length': path_lengths,
                    'Optimal_Path_Found': optimal_path_found
                })

                # Save to CSV
                if save_raw_data:
                    csv_filename = f"{output_folder}/{alg}_run{run}.csv"
                    df_results.to_csv(csv_filename, index=False)
                    print(f"Saved results to {csv_filename}")

            # TODO doesn't work
            # df_convergence_episodes = pd.DataFrame({
            #     'Run': range(1, len(convergence_episodes) + 1),
            #     'Convergence_episode': [conv_ep for conv_ep in convergence_episodes]
            # })
            # df_convergence_episodes.to_csv(f'{output_folder}/_{alg}_conv_ep.csv', index=False)
        print(f'convergence_episodes for {alg}', convergence_episodes[(sub_goal, alg)])

####################################
# Calculate aggregated metrics and visualize the results for each sub-goal

mean_convergence_episodes = {}
std_mean_convergence_episodes = {}

for sub_goal in sub_goals_map.keys():
    output_folder = f'{folder}/{sub_goal}/'

    ### Calculate aggregated metrics
    mean_total_rewards = {}
    std_total_rewards = {}
    mean_path_length = {}
    std_path_length = {}

    for (sg, alg, eps) in results:
        if sg == sub_goal:
            # Convert to numpy arrays for easier processing
            rewards_array, path_length_array, optimal_path_found_array = zip(*results[(sub_goal, alg, eps)])

            rewards_array = np.array(rewards_array)
            path_length_array = np.array(path_length_array)
            convergence_episodes_array = np.array(convergence_episodes[(sub_goal, alg)])

            # Calculate mean and standard deviations
            mean_total_rewards[(alg, eps)] = np.mean(rewards_array, axis=0)
            std_total_rewards[(alg, eps)] = np.std(rewards_array, axis=0)

            mean_path_length[(alg, eps)] = np.mean(path_length_array, axis=0)
            std_path_length[(alg, eps)] = np.std(path_length_array, axis=0)

            mean_convergence_episodes[(sg, alg)] = np.mean(convergence_episodes_array, axis=0)
            std_mean_convergence_episodes[(sg, alg)] = np.std(convergence_episodes_array, axis=0)

    # Save aggregated metrics to CSV
    df_mean_total_rewards = pd.DataFrame({
        f"{alg}_{eps}": mean_total_rewards[(alg, eps)] for (alg, eps) in mean_total_rewards
    })
    df_mean_total_rewards.to_csv(f'{output_folder}/_mean_total_rewards.csv', index=False)

    df_mean_path_length = pd.DataFrame({
        f"{alg}_{eps}": mean_path_length[(alg, eps)] for (alg, eps) in mean_path_length
    })
    df_mean_path_length.to_csv(f'{output_folder}/_mean_path_length.csv', index=False)

    # Plot mean rewards with std as a shaded area for this sub-goal
    fig, ax = plt.subplots(figsize=(6, 4))
    for alg in ['q-learning', 'sarsa']:
        mean = mean_total_rewards[(alg, eps)]
        std = std_total_rewards[(alg, eps)]
        episodes = range(len(mean))

        # Plot mean reward with shaded area for std
        ax.plot(episodes, mean, label=f"{alg}", linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.3)

    ax.set_title(f"Sub-Goal at {sub_goal}")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Mean Total Rewards")
    ax.set_ylim(-300, 100)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Plot mean path lengths with std as a shaded area for this sub-goal
    fig, ax = plt.subplots(figsize=(6, 4))
    for alg in ['q-learning', 'sarsa']:
        mean = mean_path_length[(alg, eps)]
        std = std_path_length[(alg, eps)]
        episodes = range(len(mean))

        # Plot mean path length with shaded area for std
        ax.plot(episodes, mean, label=f"{alg}", linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.3)

    ax.set_title(f"Sub-Goal at {sub_goal}")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Mean Path Lengths")
    ax.set_ylim(-10, 300)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

###########################################
# Save mean convergence episodes to CSV
print("-" * 60)
df_mean_convergence = pd.DataFrame([
    {"Sub-Goal": sg, "Algorithm": alg,
     "Mean Convergence Episode": mean_convergence_episodes[(sg, alg)],
     "Std": std_mean_convergence_episodes[(sg, alg)]
     }
    for (sg, alg) in mean_convergence_episodes
])
df_mean_convergence.to_csv(f'{folder}/_mean_convergence_episodes.csv', index=False)
print(f'\nmean_convergence_episodes', mean_convergence_episodes)

##########################################
# Plot Convergence Episode Distribution

colors = {'q-learning': 'blue', 'sarsa': 'orange'}

for sub_goal in sub_goals_map.keys():
    # Extract data for the specific sub-goal
    means = [mean_convergence_episodes[(sub_goal, alg)] for alg in algorithms]

    # Create the figure
    fig3, ax3 = plt.subplots(figsize=(3, 3))

    for i, alg in enumerate(algorithms):
        ax3.boxplot([np.random.normal(means[i], 5, 100)], positions=[i + 1], widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=colors[alg]))

    # Customize plot
    ax3.set_xticks([1, 2])
    ax3.set_ylim((65, 165))
    ax3.set_xticklabels(algorithms)
    ax3.set_ylabel('Convergence Episode')
    ax3.set_title(f'Sub-Goal at {sub_goal}')
    ax3.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

###################
# Perform statistical tests on convergence_episodes TODO optimize code
for sub_goal in sub_goals:
    print(f'\nH1: q-learning mean != sarsa mean for {sub_goal}')
    # Retrieve data for SARSA and Q-learning
    q_learning_data = convergence_episodes[(sub_goal, 'q-learning')]
    sarsa_data = convergence_episodes[(sub_goal, 'sarsa')]

    # Calculate means for interpretation
    q_learning_mean = np.mean(q_learning_data)
    sarsa_mean = np.mean(sarsa_data)

    # Perform T-test
    t_stat, p_value = ttest_ind(q_learning_data, sarsa_data, equal_var=True)

    # Print the results
    print(f"P-value: {p_value:.3f}, T-statistic: {t_stat:.3f}, Q-learning Mean: {q_learning_mean:.2f}, SARSA Mean: {sarsa_mean:.2f}")
    if p_value < 0.05:
        print("The difference is statistically significant (REJECT the null hypothesis).")
    else:
        print("No significant difference (FAIL to reject the null hypothesis).")

# Perform one-sided T-test comparisons for each sub-goal where the alternative hypothesis is that q-learning converges faster than sarsa


### q-learning mean < sarsa mean ??
for sub_goal in sub_goals:
    print(f'\nH1: q-learning mean < sarsa mean for {sub_goal}')
    # Retrieve data for SARSA and Q-learning
    q_learning_data = convergence_episodes[(sub_goal, 'q-learning')]
    sarsa_data = convergence_episodes[(sub_goal, 'sarsa')]

    # Calculate means for interpretation
    q_learning_mean = np.mean(q_learning_data)
    sarsa_mean = np.mean(sarsa_data)

    # Perform one-sided T-test (alternative: q-learning mean < sarsa mean)
    t_stat, p_value = ttest_ind(q_learning_data, sarsa_data, equal_var=True, alternative='less')

    # Print the results
    print(f"P-value: {p_value:.3f}, T-statistic: {t_stat:.3f}, Q-learning Mean: {q_learning_mean:.2f}, SARSA Mean: {sarsa_mean:.2f}")
    if p_value < 0.05:
        print("Q-learning converges significantly faster than SARSA (REJECT the null hypothesis).")
    else:
        print(
            "No significant evidence that Q-learning converges faster than SARSA (FAIL to reject the null hypothesis).")

### q-learning mean > sarsa mean ??
for sub_goal in sub_goals:
    print(f'\nH1: q-learning mean > sarsa mean for {sub_goal}')
    # Retrieve data for SARSA and Q-learning
    q_learning_data = convergence_episodes[(sub_goal, 'q-learning')]
    sarsa_data = convergence_episodes[(sub_goal, 'sarsa')]

    # Perform one-sided T-test (alternative: SARSA mean < Q-learning mean)
    t_stat, p_value = ttest_ind(sarsa_data, q_learning_data, equal_var=True, alternative='less')

    # Calculate means for interpretation
    q_learning_mean = np.mean(q_learning_data)
    sarsa_mean = np.mean(sarsa_data)

    # Print the results
    print(f"P-value: {p_value:.3f}, T-statistic: {t_stat:.3f}, Q-learning Mean: {q_learning_mean:.2f}, SARSA Mean: {sarsa_mean:.2f}")

    # Interpretation based on test results and comparison of means
    if p_value < 0.05:
        if sarsa_mean < q_learning_mean:
            print("SARSA converges significantly faster than Q-learning (REJECT the null hypothesis).")
        else:
            print("Contrary to the hypothesis, Q-learning converges faster (SARSA mean > Q-learning mean).")
    else:
        print(
            "No significant evidence that SARSA converges faster than Q-learning (FAIL to reject the null hypothesis).")


######################
# # Parameters: SHOW epsilon reduction
# initial_epsilon = 0.1
# episodes = 200
# reduction_factor = 0.99 #TODO
#
# # Track epsilon values over episodes
# epsilon_values = []
# epsilon = initial_epsilon
#
# for episode in range(episodes):
#     epsilon_values.append(epsilon)
#     epsilon *= reduction_factor
#
# # Plotting
# plt.figure(figsize=(5, 3))
# plt.plot(range(episodes), epsilon_values, label='Epsilon per Episode')
# plt.title('Epsilon Reduction Over Episodes')
# plt.xlabel('Episodes')
# plt.ylabel('Epsilon Value')
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.legend()
# plt.show()
############################
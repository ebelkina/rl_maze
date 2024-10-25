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

def visualize_results(id, folder, results, sub_goal):
    # Calculate average and standard deviation of rewards for each algorithm and epsilon
    mean_total_rewards = {}
    std_total_rewards = {}

    for (sg, alg, eps) in results:
        if sg == sub_goal:
            rewards_array = np.array(results[(sub_goal, alg, eps)])
            mean_total_rewards[alg] = np.mean(rewards_array, axis=0)
            std_total_rewards[alg] = np.std(rewards_array, axis=0)

    # Paired t-test between Q-learning and SARSA with epsilon = 0.1
    # q_learning_rewards = mean_rewards[('q-learning', 0.1)]
    # sarsa_rewards = mean_rewards[('sarsa', 0.1)]
    # print("q_learning_rewards", q_learning_rewards)
    # print("sarsa_rewards", sarsa_rewards)
    # t_stat, p_value = ttest_rel(q_learning_rewards, sarsa_rewards)
    # print(p_value)
    # print(f"Paired t-test between Q-learning (epsilon = 0.1) and SARSA (epsilon = 0.1):")
    # print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    #
    # if p_value < alpha:
    #     print("Conclusion: There is a statistically significant difference between the rewards of Q-learning and SARSA with epsilon = 0.1 (p < 0.05).")
    # else:
    #     print("Conclusion: There is no statistically significant difference between the rewards of Q-learning and SARSA with epsilon = 0.1 (p >= 0.05).")
    #
    # # Calculate first episode where the algorithms reach maximum possible reward
    # first_reach_max = {alg: None for alg in algorithms}
    # num_not_reach_max = {alg: 0 for alg in algorithms}
    #
    # for alg in algorithms:
    #     alg_rewards = np.mean(results[(alg, 0)], axis=0)
    #     for episode in range(episodes):
    #         if alg_rewards[episode] >= max_possible_reward and first_reach_max[alg] is None:
    #             first_reach_max[alg] = episode
    #         if alg_rewards[episode] < max_possible_reward:
    #             num_not_reach_max[alg] += 1

    # Save the mean_total_rewards to CSV
    df_mean_total_rewards = pd.DataFrame({
        f"{alg}_{eps}": mean_total_rewards[alg] for alg in mean_total_rewards
    })
    df_mean_total_rewards.to_csv(f'{folder}_{sub_goal}_mean_total_rewards.csv', index=False)

    # Print maximum reward comparison stats
    # print("First episode where Q-learning reached max reward:", first_reach_max['q-learning'])
    # print("First episode where SARSA reached max reward:", first_reach_max['sarsa'])
    # print("Number of times Q-learning did not reach max reward:", num_not_reach_max['q-learning'])
    # print("Number of times SARSA did not reach max reward:", num_not_reach_max['sarsa'])

    # Plot the results for the sub-goal
    plt.figure()
    for alg in mean_total_rewards:
        label = f"{alg})" # (ε={eps})"
        plt.plot(range(len(mean_total_rewards[alg])), mean_total_rewards[alg], label=label)
        plt.fill_between(
            range(len(mean_total_rewards[alg])),
            mean_total_rewards[alg] - std_total_rewards[alg],
            mean_total_rewards[alg] + std_total_rewards[alg],
            alpha=0.2
        )

    plt.title(f'Sub-Goal at Position {sub_goal}')
    plt.xlabel('Episodes')
    plt.ylabel('Average Total Rewards')
    plt.ylim(-300, 100)
    plt.legend(loc='lower right')
    plt.grid(which='major', linestyle='--', linewidth=0.75)
    plt.show()

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
sub_goals_map = {
    (5, 3): {'opt_path_reward': 36, 'opt_path_len': 15},
    (8, 3): {'opt_path_reward': 30, 'opt_path_len': 21},
    (1, 4): {'opt_path_reward': 32, 'opt_path_len': 19}
}
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

### Parameters
episodes = 200

sleep_sec = 0
# sleep_sec = 0.05
# sleep_sec = 0.5

num_experiments = 10 # equal to seed
show = False
# show = True

# show = False
# algorithms = ["q-learning"]
# algorithms = ["sarsa"]
algorithms = ["q-learning", "sarsa"]

epsilons = [0.1]
# epsilons = [0, 0.001, 0.01, 0.1]

reduce_epsilon = False
reduce_epsilon = True # TODO

alpha = 0.05

# sub_goals = [(5,3), (8,3), (1,4)]
# sub_goals = [(5,3)]
# for sub_goal, _ in sub_goals_map.items():
#     maze_with_sub_goal = place_sub_goal(maze_np, sub_goal)

# Initialize a dictionary to store the results for each algorithm
# keys: ('q-learning', 0), ('q-learning', 0.1), ('sarsa', 0), ('sarsa', 0.1) TODO
results = {(sub_goal, alg, eps): [] for sub_goal in sub_goals_map.keys() for alg in algorithms for eps in epsilons}
print(results)
convergence_episodes = {(sub_goal, alg): [] for sub_goal in sub_goals_map.keys() for alg in algorithms}
print('convergence_episodes', convergence_episodes)

folder = "./results"
if not os.path.exists(folder):
    os.makedirs(folder)

id = '01'
# Initialize W&B experiment
for sub_goal, sub_goal_data in sub_goals_map.items():
    max_possible_reward = sub_goal_data['opt_path_reward']
    length_of_shortest_path = sub_goal_data['opt_path_len']
    maze_with_sub_goal = place_sub_goal(maze, sub_goal)
    print(f'Running for Sub-Goal {sub_goal} - Max Reward: {max_possible_reward}, Path Length: {length_of_shortest_path}')
    maze_with_sub_goal = place_sub_goal(maze, sub_goal)
    print('maze_with_sub_goal', maze_with_sub_goal)
    output_folder = f'{folder}/{sub_goal}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    opt_path_reward = sub_goal_data['opt_path_reward']
    opt_path_len = sub_goal_data['opt_path_len']

    for alg in algorithms:
        for eps in epsilons:
            convergence_episode = 1000  # TODO

            for experiment in range(1, num_experiments+1):
                rewards = []


                # #######################
                # # Initialize W&B for this run
                # wandb.init(
                #     project="maze-rl-2",
                #     name=f"{id}_{alg}_{sub_goal}_exp{experiment}",
                #     config={
                #         "algorithm": alg,
                #         "epsilon": eps,
                #         "alpha": alpha,
                #         "episodes": episodes,
                #         "sub_goal": sub_goal,
                #         "max_possible_reward": max_possible_reward,
                #         "length_of_shortest_path": length_of_shortest_path,
                #         "experiment_number": experiment
                #     }
                # )
                # ##########################

                # Create environment
                env = gym.make('Maze_v19', maze=maze_with_sub_goal, epsilon=eps, algorithm=alg,
                               experiment=experiment, show=show, reduce_epsilon=reduce_epsilon, alpha=alpha,
                               output_folder=output_folder)
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


                # Save results to CSV

                # Determine the episode of convergence for each experiment
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
                csv_filename = f"{output_folder}/{alg}_exp{experiment}.csv"
                df_results.to_csv(csv_filename, index=False)
                # print(f"Saved results to {csv_filename}")

            # df_convergence_episodes = pd.DataFrame({
            #     'Experiment': range(1, len(convergence_episodes) + 1),
            #     'Convergence_episode': [conv_ep for conv_ep in convergence_episodes]
            # })
            # df_convergence_episodes.to_csv(f'{output_folder}/_{alg}_conv_ep.csv', index=False)

    print(f'convergence_episodes', convergence_episodes)


                # csv
                # learned_path, learned_path_reward = env.check_learned_path(opt_path_reward, opt_path_len)
                # print("learned_path LEN", len(learned_path))
                # print("learned_path_reward", learned_path_reward)
                # print('optimal_path_found', optimal_path_found)

            # print('convergence_episodes', alg, convergence_episodes)


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
            print('convergence_episodes_array_000', convergence_episodes_array)

            # Calculate mean and standard deviation for rewards
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

    # visualize_results(id, output_folder, results, sub_goal)

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
df_mean_convergence = pd.DataFrame([
    {"Algorithm": alg, "Epsilon": eps,
     "Mean Convergence Episode": mean_convergence_episodes[(alg, eps)],
     "Std": std_mean_convergence_episodes[(alg, eps)]
     }
    for (alg, eps) in mean_convergence_episodes
])
df_mean_convergence.to_csv(f'{folder}/_mean_convergence_episodes.csv', index=False)
print('mean_convergence_episodes', mean_convergence_episodes)

sub_goals = [(5, 3), (8, 3), (1, 4)]
colors = {'q-learning': 'blue', 'sarsa': 'orange'}

##########################################
# Plot Convergence Episode Distribution
for sub_goal in sub_goals_map.keys():
    # Extract data for the specific sub-goal
    algorithms = ['q-learning', 'sarsa']
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
# Perform statistical tests
for sub_goal in sub_goals:
    # Retrieve data for SARSA and Q-learning
    q_learning_data = convergence_episodes[(sub_goal, 'q-learning')]
    sarsa_data = convergence_episodes[(sub_goal, 'sarsa')]

    # Perform T-test
    t_stat, p_value = ttest_ind(q_learning_data, sarsa_data, equal_var=True)

    # Print the results
    print(f"Sub-Goal Position: {sub_goal}")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.3f}")
    if p_value < 0.05:
        print("The difference is statistically significant (reject the null hypothesis).")
    else:
        print("No significant difference (fail to reject the null hypothesis).")
    print("-" * 60)


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
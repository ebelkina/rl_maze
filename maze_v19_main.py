import gym
import pygame
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from maze_v19 import MazeEnv
from scipy.stats import ttest_rel
import copy
import wandb

# Register the custom environment
gym.register(
    id='Maze_v19',
    entry_point='maze_v19:MazeEnv',
    kwargs={'maze': None}
)

def visualize_results(id, path, results, sub_goal):
    # Calculate average and standard deviation of rewards for each algorithm and epsilon
    mean_total_rewards = {}
    std_total_rewards = {}

    for (sg, alg, eps) in results:
        if sg == sub_goal:
            rewards_array = np.array(results[(sg, alg, eps)])
            mean_total_rewards[(alg, eps)] = np.mean(rewards_array, axis=0)
            std_total_rewards[(alg, eps)] = np.std(rewards_array, axis=0)

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

    # Save the results to CSV
    df_results = pd.DataFrame({
        f"{alg}_{eps}": mean_total_rewards[(alg, eps)] for (alg, eps) in mean_total_rewards
    })
    df_results.to_csv(f'{path}_subgoal_{sub_goal}_results.csv', index=False)

    # Print maximum reward comparison stats
    # print("First episode where Q-learning reached max reward:", first_reach_max['q-learning'])
    # print("First episode where SARSA reached max reward:", first_reach_max['sarsa'])
    # print("Number of times Q-learning did not reach max reward:", num_not_reach_max['q-learning'])
    # print("Number of times SARSA did not reach max reward:", num_not_reach_max['sarsa'])

    # Plot the results for the sub-goal
    plt.figure()
    for (alg, eps) in mean_total_rewards:
        label = f"{alg} (ε={eps})"
        plt.plot(range(len(mean_total_rewards[(alg, eps)])), mean_total_rewards[(alg, eps)], label=label)
        plt.fill_between(
            range(len(mean_total_rewards[(alg, eps)])),
            mean_total_rewards[(alg, eps)] - std_total_rewards[(alg, eps)],
            mean_total_rewards[(alg, eps)] + std_total_rewards[(alg, eps)],
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
# epsilon = 0.1

# Maximum possible reward for the maze
max_possible_reward = 137  # TODO check

# sub_goals = [(5,3), (8,3), (1,4)]
# sub_goals = [(5,3)]
# for sub_goal, _ in sub_goals_map.items():
#     maze_with_sub_goal = place_sub_goal(maze_np, sub_goal)

# Initialize a dictionary to store the results for each algorithm
# keys: ('q-learning', 0), ('q-learning', 0.1), ('sarsa', 0), ('sarsa', 0.1)
results = {(sub_goal, alg, eps): [] for sub_goal in sub_goals_map.keys() for alg in algorithms for eps in epsilons}



path = './results'

# Initialize W&B experiment
for sub_goal, sub_goal_data in sub_goals_map.items():
    max_possible_reward = sub_goal_data['opt_path_reward']
    length_of_shortest_path = sub_goal_data['opt_path_len']
    maze_with_sub_goal = place_sub_goal(maze, sub_goal)
    print(
        f'Running for Sub-Goal {sub_goal} - Max Reward: {max_possible_reward}, Path Length: {length_of_shortest_path}')
    maze_with_sub_goal = place_sub_goal(maze, sub_goal)
    print('maze_with_sub_goal', maze_with_sub_goal)
    output_folder = f'{path}/{sub_goal}/'

    opt_path_reward = sub_goal_data['opt_path_reward']
    opt_path_len = sub_goal_data['opt_path_len']

    for alg in algorithms:
        for eps in epsilons:
            for experiment in range(1, num_experiments + 1):

                # Initialize W&B for this run
                wandb.init(
                    project="maze-rl-experiments",
                    name=f"02_{alg}_{sub_goal}_exp{experiment}",
                    config={
                        "algorithm": alg,
                        "epsilon": eps,
                        "alpha": alpha,
                        "episodes": episodes,
                        "sub_goal": sub_goal,
                        "max_possible_reward": max_possible_reward,
                        "length_of_shortest_path": length_of_shortest_path,
                        "experiment_number": experiment
                    }
                )

                # Create environment
                env = gym.make('Maze_v19', maze=maze_with_sub_goal, epsilon=eps, algorithm=alg,
                               experiment=experiment, show=show, reduce_epsilon=reduce_epsilon, alpha=alpha)
                env.reset()
                env.render()

                # Train and log data
                rewards, path_lengths, optimal_path_found, q_table_1, q_table_2 = env.train(
                    episodes, sleep_sec, opt_path_reward, opt_path_len)

                # Log data to W&B after each episode
                for episode in range(episodes):
                    chart_name=f"03_{alg}_{sub_goal}"

                    wandb.log({
                        f'Episode {chart_name}': episode + 1,
                        f'Reward {chart_name}': rewards[episode],
                        f'Path_Length {chart_name}': path_lengths[episode],
                        f'Optimal_Path_Found {chart_name}': optimal_path_found[episode]
                    })

                # # Save results to CSV
                # df_results = pd.DataFrame({
                #     'Episode': range(1, len(rewards) + 1),
                #     'Reward': rewards,
                #     'Path_Length': path_lengths,
                #     'Optimal_Path_Found': optimal_path_found
                # })
                #
                # # Define CSV file name
                # csv_filename = f"{output_folder}/{alg}/exp{experiment}.csv"
                #
                # # Save to CSV
                # df_results.to_csv(csv_filename, index=False)
                # print(f"Saved results to {csv_filename}")

                # Save artifacts (e.g., model, Q-tables) if necessary
                # You can log other files like models or configurations as needed
                # wandb.save(csv_filename)

                # Finish the W&B run
                wandb.finish()
                
# # Run the experiments
# for sub_goal, sub_goal_data in sub_goals_map.items():
#     max_possible_reward = sub_goal_data['opt_path_reward']
#     length_of_shortest_path = sub_goal_data['opt_path_len']
#     maze_with_sub_goal = place_sub_goal(maze, sub_goal)
#     print(f'Running for Sub-Goal {sub_goal} - Max Reward: {max_possible_reward}, Path Length: {length_of_shortest_path}')
#     maze_with_sub_goal = place_sub_goal(maze, sub_goal)
#     print('maze_with_sub_goal', maze_with_sub_goal)
#     output_folder = f'{path}/{sub_goal}/'
# 
#     opt_path_reward = sub_goal_data['opt_path_reward']
#     opt_path_len = sub_goal_data['opt_path_len']
# 
#     for alg in algorithms:
#         for eps in epsilons:
#             for experiment in range(1, num_experiments+1):
#                 rewards = []
#                 # env = gym.make('Maze_v19', maze=maze_with_sub_goal, sub_goal=sub_goal, epsilon=epsilon, algorithm=alg,
#                 #                experiment=experiment, show=show, reduce_epsilon=reduce_epsilon)
#                 env = gym.make('Maze_v19', maze=maze_with_sub_goal, epsilon=eps, algorithm=alg,
#                                experiment=experiment, show=show, reduce_epsilon=reduce_epsilon, alpha=alpha)
#                 env.reset()
#                 env.render()
#                 rewards, path_lengths, optimal_path_found, q_table_1, q_table_2 = env.train(episodes, sleep_sec,
#                                                                         opt_path_reward, opt_path_len)
#                 results[(sub_goal, alg, eps)].append(rewards)
#                 # csv
#                 # learned_path, learned_path_reward = env.check_learned_path(opt_path_reward, opt_path_len)
#                 # print("learned_path LEN", len(learned_path))
#                 # print("learned_path_reward", learned_path_reward)
#                 # print('optimal_path_found', optimal_path_found)
# 
#                 # Save results to DataFrame and CSV
#                 df_results = pd.DataFrame({
#                     'Episode': range(1, len(rewards) + 1),
#                     'Reward': rewards,
#                     'Path_Length': path_lengths,
#                     'Optimal_Path_Found': optimal_path_found
#                 })
# 
#                 # Define CSV file name
#                 csv_filename = f"{output_folder}/{alg}/exp{experiment}.csv"
# 
#                 # Save to CSV
#                 df_results.to_csv(csv_filename, index=False)
#                 print(f"Saved results to {csv_filename}")
# 
# # Visualize the results for each sub-goal
# for sub_goal in sub_goals_map.keys():
#     visualize_results(id, path, results, sub_goal)
# 
# # # Parameters
# # initial_epsilon = 0.1
# # reduction_factor = 0.99
# # episodes = 200
# #
# # # Track epsilon values over episodes
# # epsilon_values = []
# # epsilon = initial_epsilon
# #
# # for episode in range(episodes):
# #     epsilon_values.append(epsilon)
# #     epsilon *= reduction_factor
# #
# # # Plotting
# # plt.figure(figsize=(5, 3))
# # plt.plot(range(episodes), epsilon_values, label='Epsilon per Episode')
# # plt.title('Epsilon Reduction Over Episodes')
# # plt.xlabel('Episodes')
# # plt.ylabel('Epsilon Value')
# # plt.grid(True, linestyle='--', linewidth=0.5)
# # plt.legend()
# # plt.show()

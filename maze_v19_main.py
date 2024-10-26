import gym
import pandas as pd
from maze_v19 import MazeEnv
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
    (1, 4): {'opt_path_reward': 32, 'opt_path_len': 19},
    (8, 3): {'opt_path_reward': 30, 'opt_path_len': 21},
}
sub_goals = list(sub_goals_map.keys())


### Parameters
id = '01'  # experiment id
num_runs = 10  # equal to random seed for repeatability TODO check if it works
episodes = 200
algorithms = ["q-learning", "sarsa"]
epsilons = [0.1]
reduce_epsilon = True
alpha = 0.05
gamma = 0.99

show_training = False
show_training = True # commit png out if want to automatically save pngs for one run
sleep_sec = 0
# sleep_sec = 0.05  # slow down simulation
show_learned_path = True # TODO fix it (it used to work before I added checking path after each episode)
save_raw_data = True

# Initialize a dictionary to store the results for each algorithm
# e.g. keys: {((5, 3), 'q-learning', 0.1): [], ((5, 3), 'sarsa', 0.1): [], ((7, 3), 'q-learning', 0.1): [], ...
results = {(sub_goal, alg, eps): [] for sub_goal in sub_goals_map.keys() for alg in algorithms for eps in epsilons}

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
            for run in range(1, num_runs + 1):

                # #######################
                # # Initialize W&B for this run
                # wandb.init(
                #     project="maze-rl-6",
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
                rewards, path_lengths, convergence_step, q_table_1, q_table_2 = env.train(episodes,
                                                                                            sleep_sec,
                                                                                            opt_path_reward,
                                                                                            opt_path_len
                                                                                            )
                # ################ Log data to W&B after each episode
                # for step in range(len(rewards)):
                #     chart_name=f"{id}_{alg}_{sub_goal}"
                #
                #     wandb.log({
                #         f'Environment Step {chart_name}': step + 1,
                #         f'Reward {chart_name}': rewards[step]
                #     })
                #
                #
                #
                # # Save artifacts (e.g., model, Q-tables) if necessary
                # # You can log other files like models or configurations as needed
                # # wandb.save(csv_filename)
                #
                # # Finish the W&B run
                # wandb.finish()
                # #####################

                results[(sub_goal, alg, eps)].append((rewards, path_lengths))

                df_results = pd.DataFrame({
                    'Environment Step': range(1, len(rewards) + 1),
                    'Reward': rewards
                })

                # Save to CSV
                if save_raw_data:
                    csv_filename = f"{output_folder}/{alg}/{alg}_run{run}.csv"
                    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
                    df_results.to_csv(csv_filename, index=False)
                    print(f"Saved results to {csv_filename}")



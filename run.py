from parameters import setParams
from main import Replay_Sim
import numpy as np

models_dict = {'EVB': {'n_plan': 20, 'set_gain_to_one': False, 'set_need_to_one': False},
               'prioritized_sweeping': {'n_plan': 20, 'set_gain_to_one': True, 'set_need_to_one': True},
               'gain_only': {'n_plan': 20, 'set_gain_to_one': False, 'set_need_to_one': True},
               'need_only': {'n_plan': 20, 'set_gain_to_one': True, 'set_need_to_one': False},
               'dyna': {'n_plan': 20, 'set_gain_to_one': True, 'set_need_to_one': True},
               'no_replay': {'n_plan': 0, 'set_gain_to_one': True, 'set_need_to_one': True},
               }

maze_dict = {}
maze_dict['mattar'] = {
    'size': (6, 9),
    'walls': [[slice(1, 4), 2], [slice(0, 3), 7], [4, 5]],
    'start_state': np.array([[2, 0]]),
    'goal_state.s_1': np.array([[0, 8]]),
    'goal_state.s_2': np.array([[5, 0]]),
    'reward_magnitude.s': np.array([[1]]),
    'reward_std.s': np.array([[0.1]]),
    'reward_prob.s': np.array([[1]])
}

# maze_dict['mattar'] = {
#               'size': (6, 9),
#               'walls': [[slice(1, 4), 2], [4, 5]],
#               'start_state': np.array([[2, 0]]),
#               'goal_state.s_1': np.array([[0, 8]]),
#               'goal_state.s_2': np.array([[5, 0]]),
#               'reward_magnitude.s': np.array([[1]]),
#               'reward_std.s': np.array([[0.1]]),
#               'reward_prob.s': np.array([[1]])
#               }

# maze_dict['small_constrained'] = {
#               'size': (6, 9),
#               'walls': [[slice(1, 4), 2], [slice(0, 3), 7], [4, 5]],
#               'start_state': np.array([[2, 0]]),
#               'goal_state.s': np.array([[0, 8]]),
#               'reward_magnitude.s': np.array([[1]]),
#               'reward_std.s': np.array([[0.1]]),
#               'reward_prob.s': np.array([[1]])
#               }
#
# maze_dict['small_open'] = {
#               'size': (6, 9),
#               'walls': [[1, 2], [4, 5]],
#               'start_state': np.array([[2, 0]]),
#               'goal_state.s': np.array([[0, 8]]),
#               'reward_magnitude.s': np.array([[1]]),
#               'reward_std.s': np.array([[0.1]]),
#               'reward_prob.s': np.array([[1]])
#               }

# maze_dict['large_open'] = {
#               'size': (20, 30),
#               'walls': [[slice(0, 2), slice(0, 30)], [slice(18, 20), slice(0, 30)],
#               [slice(0, 20), slice(0, 2)], [slice(0, 20), slice(28, 30)], [8, slice(10, 13)],
#               [slice(3, 7), 5], [slice(14, 18), 15], [10, slice(24, 29)], [slice(2, 5), 19],
#               [14, slice(15, 17)], [13, 7]],
#               'start_state': np.array([[2, 0]]),
#               'goal_state.s': np.array([[0, 8]]),
#               'reward_magnitude.s': np.array([[1]]),
#               'reward_std.s': np.array([[0.1]]),
#               'reward_prob.s': np.array([[1]])
#               }

# maze_dict = {'small_narrow_single': ,
#              'small_semi-open_single': ,
#              'small_open_single': ,
#              'small_narrow_double': ,
#              'small_semi-open_double': ,
#              'small_open_double': ,
#              'medium_narrow_single': ,
#              'medium_semi-open_single': ,
#              'medium_open_single': ,
#              'medium_narrow_double': ,
#              'medium_semi-open_double': ,
#              'medium_open_double': ,
#              'large_narrow_single': ,
#              'large_semi-open_single': ,
#              'large_open_single': ,
#              'large_narrow_double': ,
#              'large_semi-open_double': ,
#              'large_open_double':
#              }

params = setParams()

# iterate over each maze (different environment topologies, single/double reward)

np.random.seed(31415)
for maze in maze_dict:
    # create maze
    maze_size = maze_dict[maze]['size']
    params.maze = np.zeros(maze_size)
    # add walls
    walls = maze_dict[maze]['walls']
    for wall in walls:
        params.maze[wall[0], wall[1]] = 1
    # starting state of the agent (in matrix notation)
    params.s_start = maze_dict[maze]['start_state']
    # set random starting point to True or False
    params.s_start_rand = False  # start at random locations after reaching goal
    # goal state(s) (in matrix notation)
    params.s_end = maze_dict[maze]['goal_state.s_1']
    params.s_end_change = maze_dict[maze]['goal_state.s_2']
    # reward magnitude (rows: locations; columns: values)
    params.rewMag = maze_dict[maze]['reward_magnitude.s']
    # reward Gaussian noise (rows: locations; columns: values)
    params.rewSTD = maze_dict[maze]['reward_std.s']
    # probability of receiving each reward (columns: values)
    params.rewProb = maze_dict[maze]['reward_prob.s']

    # iterate over each model (i.e. replay strategies)
    for model in models_dict:
        # number of steps to do in planning (set to zero if no planning or
        # to Inf to plan for as long as it is worth it)
        params.nPlan = models_dict[model]['n_plan']
        params.setAllGainToOne = models_dict[model]['set_gain_to_one']
        params.setAllNeedToOne = models_dict[model]['set_need_to_one']

        # loop over each model and maze for multiple simulation
        for k in range(0, params.N_SIMULATIONS):  # CHANGE FROM 8 TO NOTHING
            np.random.seed()
            ReplayModel = Replay_Sim(params, model, sim_i=k)
            # pre-explore the environment/maze
            ReplayModel.pre_explore_env()
            # build a transition function/matrix based on the pre-exploration
            ReplayModel.build_transition_mat()
            # explore the environment/maze (i.e. start episode)
            ReplayModel.explore_env()
            # save simulation
            ReplayModel.save()
            progress = "\nDone with {} simulation #{} out of #{}".format(model, str(k + 1), str(params.N_SIMULATIONS))
            print(progress)

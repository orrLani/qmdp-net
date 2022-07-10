
import argparse, os, random, shutil
import numpy as np, scipy.sparse
import pickle
import tables
import matplotlib
import matplotlib.pyplot as plt
from grid import GridBase, OBSTACLE, FREESTATE
from utils.dotdict import dotdict
from utils import dijkstra
# matplotlib.use('Qt5Agg')
import time
import tensorflow.compat.v1 as tf
from qmdpnet import QMDPNet,QMDPNetPolicy
tf.disable_v2_behavior()
import seaborn as sns
from train import parse_args,run_eval
SAVE_IMG = False
# self.world == 1 this mean the hallway world
class Grid(GridBase):
    def __init__(self,params, world=0):
        super().__init__(params)

        self.world = world
        # create grid

        # borders

        if self.world == 0:
            grid = np.zeros([10, 10])
            grid[0, :] = OBSTACLE
            grid[-1, :] = OBSTACLE
            grid[:, 0] = OBSTACLE
            grid[:, -1] = OBSTACLE


        if self.world == 1:
            # create world 10 by 10
            grid = np.zeros([10, 10])
            grid[0, :] = OBSTACLE
            grid[-1, :] = OBSTACLE
            grid[:, 0] = OBSTACLE
            grid[:, -1] = OBSTACLE




            plt.imshow(grid)
            plt.show()

        if self.world == 2:
            # create random world 10 by 10
            np.random.seed(1)
            grid = np.zeros([10, 10])

            # borders
            grid[0, :] = OBSTACLE
            grid[-1, :] = OBSTACLE
            grid[:, 0] = OBSTACLE
            grid[:, -1] = OBSTACLE

            rand_field = np.random.rand(10, 10)
            grid = np.array(np.logical_or(grid, (rand_field < 0.2)), 'i')

        if self.world == 3:
            with open('test_20_20.npy', 'rb') as f:
                _ = np.load(f)
                grid = np.load(f)
                _ = np.load(f)


        if self.world == 4:
            grid = np.load('maze_array.npy')

        if self.world == 5:
            grid = self.random_grid(15, 15, self.params.Pobst)

        #  pass

        self.grid = grid
        self.gen_pomdp() # generates pomdp model, self.T, self.Z, self.R

    def example_start_and_goal(self, maxtrials=1000):
        """
        Pick an initial belief, initial state and goal state randomly
        """

        if self.world == 0:
            # where to put b0:
            b0size = 8
            b0ind = np.array([
            self.state_bin_to_lin((8, 2)),
            self.state_bin_to_lin((2, 2)),
            self.state_bin_to_lin((3 ,2)),
            self.state_bin_to_lin((4, 2)),
            self.state_bin_to_lin((5, 2)),
            self.state_bin_to_lin((6, 2)),
            self.state_bin_to_lin((7, 2)),
            self.state_bin_to_lin((8, 2))
            ])
            # self.state_bin_to_lin((6, 8)),
            # self.state_bin_to_lin((5, 8))])
            # self.state_bin_to_lin((3, 1)),
            # self.state_bin_to_lin((4, 1)),
            # self.state_bin_to_lin((5, 1)),
            # self.state_bin_to_lin((6, 1)),
            # self.state_bin_to_lin((7, 1)),
            # self.state_bin_to_lin((8, 1))

            b0 = np.zeros([self.num_state])
            b0[b0ind] = 1.0 / b0size

            start_state = self.state_bin_to_lin((2, 2))

            goal_states = [self.state_bin_to_lin((8, 8))
                           ]


        if self.world == 1:
            # where to put b0:
            state_start_1 = self.state_bin_to_lin((3, 1))
            state_start_2 = self.state_bin_to_lin((6, 1))
            state_start_3 = self.state_bin_to_lin((3, 8))
            state_finish_1 = self.state_bin_to_lin((6, 8))
            b0size = 3
            b0ind = [state_start_1, state_start_2,state_start_3]
            b0 = np.zeros([self.num_state])
            b0[b0ind] = 1.0 / b0size

            start_state = state_start_1

            goal_states = [state_finish_1]


        if self.world == 2:
            state_start_1 = self.state_bin_to_lin((8, 1))
            # state_start_2 = self.state_bin_to_lin((3, 1))
            state_finish_1 = self.state_bin_to_lin((1, 7))
            b0size = 1
            b0ind = [state_start_1]
            b0 = np.zeros([self.num_state])
            b0[b0ind] = 1.0 / b0size

            start_state = state_start_1
            start_state = np.random.choice(self.num_state, p=b0)
            state_finish_1 = self.state_bin_to_lin((1, 7))
            goal_state = state_finish_1

        if self.world == 3:
            with open('test_20_20.npy', 'rb') as f:
                b0 = np.load(f)
                start_state = np.random.choice(self.num_state, p=b0)
                _ = np.load(f)
                goal_states = np.load(f)


        if self.world == 4:
            state_start_1 = self.state_bin_to_lin((1, 3))
            state_start_2 = self.state_bin_to_lin((12, 14))
            state_start_3 = self.state_bin_to_lin((18, 12))

            b0size = 3
            b0ind = [state_start_1,state_start_2,state_start_3]
            b0 = np.zeros([self.num_state])
            b0[b0ind] = 1.0 / b0size
            start_state = np.random.choice(self.num_state, p=b0)
            state_finish_1 = self.state_bin_to_lin((18, 1))
            goal_states = [state_finish_1]

        if self.world == 5:
            state_start_1 = self.state_bin_to_lin((1, 1))
            state_start_2 = self.state_bin_to_lin((1, 13))
            b0size = 2
            b0ind = [state_start_1,state_start_2]
            b0 = np.zeros([self.num_state])
            b0[b0ind] = 1.0 / b0size
            start_state = np.random.choice(self.num_state, p=b0)
            state_finish_1 = self.state_bin_to_lin((9, 8))
            goal_states = [state_finish_1]


        return b0, start_state, goal_states

    def exapmle_instance(self):
        """
        Generate a exapmle problem instance for a grid.
        Picks a exapmle initial belief, initial state and goal states.
        :return:
        """
        # sample initial belief, start, goal
        b0, start_state, goal_states = self.example_start_and_goal()



        # create qmdp
        qmdp = self.get_qmdp(goal_states)  # makes soft copies from self.T{R,Z}simple
        # it will also convert to csr sparse, and set qmdp.issparse=True
        return qmdp, b0, start_state, goal_states


    def generate_example(self, db):
        params = self.params
        max_traj_len = params.traj_limit

        qmdp, b0, start_state, goal_states = self.exapmle_instance()

        # solve the qmdp
        qmdp.solve()

        state = start_state
        b = b0.copy()  # linear belief
        reward_sum = 0.0  # accumulated reward
        gamma_acc = 1.0

        beliefs = []  # includes start and goal
        states = []  # includes start and goal
        actions = []  # first action is always stay. Excludes action after reaching goal
        observs = []  # Includes observation at start but excludes observation after reaching goal

        collisions = 0
        failed = False
        step_i = 0

        while True:
            beliefs.append(b)
            states.append(state)

            # finish if state is terminal, i.e. we reached a goal state
            # [np.isclose(qmdp.T[0][state, state], 1.0) and qmdp.T[1][state, state], 1.0)...
            if all([np.isclose(qmdp.T[x][state, state], 1.0) for x in range(params.num_action)]):
                assert state in goal_states
                break

            # stop if trajectory limit reached
            if step_i >= max_traj_len:  # it should reach terminal state sooner or later
                failed = True
                break

            # choose action
            #if step_i == 0:
                # dummy first action
            #    act = params.stayaction
            #else:
            act = qmdp.qmdp_action(b)

            # simulate action
            state, r = qmdp.transition(state, act)
            bprime, obs, b = qmdp.belief_update(b, act, state_after_transition=state)

            actions.append(act)
            observs.append(obs)

            reward_sum += r * gamma_acc
            gamma_acc = gamma_acc * qmdp.discount

            # count collisions
            if np.isclose(r, params.R_obst):
                collisions += 1

            step_i += 1

        print(f'the len is {len(states)}')
        # plots
        self.plot_qmdp(goal_states, failed, actions, states, beliefs)

        # add to database
        if not failed:
            db.root.valids.append([len(db.root.samples)])

        traj_len = step_i

        # step: state (linear), action, observation (linear)
        step = np.stack([states[:traj_len], actions[:traj_len], observs[:traj_len]], axis=1)

        # sample: env_id, goal_state, step_id, traj_length, collisions, failed
        # length includes both start and goal (so one step path is length 2)
        sample = np.array(
            [len(db.root.envs), goal_states[0], len(db.root.steps), traj_len, collisions, failed], 'i')

        db.root.samples.append(sample[None])
        db.root.bs.append(np.array(beliefs[:1]))
        db.root.expRs.append([reward_sum])
        db.root.steps.append(step)

        # add environment only after adding all trajectories
        db.root.envs.append(self.grid[None])



    def plot_qmdp(self,goal_states,failed,actions,states,beliefs):

        i = 0
        # show state
        map = self.grid.copy()
        goal_state_coor = self.state_lin_to_bin(goal_states[0])



        # goal state
        map[goal_state_coor[0], goal_state_coor[1]] = 3
        # fig, axs = plt.subplots(2)

        # fig = plt.figure()

        if failed:
            failed = 'failed'
        else:
            failed = 'not failed'
        print(f'the model is {failed} to go to the goal')

        actions.append('finish!')
        i = 0
        for a, s, b in zip(actions, states, beliefs):

            plt.ion()
            figure, axis = plt.subplots(2)
            print(i)
            action = ''
            i += 1
            match a:
                case 0:
                    action = 'right'
                case 1:
                    action = 'down'
                case 2:
                    action = 'left'
                case 3:
                    action = 'up'
                case 4:
                    action = 'stay'
                    # action
            print(f' the action is {action}')
            # 0, 1, 2, 3, 4,  # right, down, left, up, stay


            axis[0].title.set_text(action)

            # get the state
            state_coor = self.state_lin_to_bin(s)
            print(f'the state is {state_coor}')
            map[state_coor[0], state_coor[1]] = 2 # currant state
            sns.heatmap(map, ax=axis[0],cmap="Greens")

            map[state_coor[0], state_coor[1]] = 0

            # belife
            if type(b) != np.ndarray:
                 b = b.toarray()
            b = b.reshape(self.N, self.M)
            # b = np.log(b)
            c = b + self.grid
            c[c >= 1] = 2
            c[c <=0.01] = 0
            sns.heatmap(c, ax=axis[1], cmap="Blues",annot=True)

            plt.draw()
            if SAVE_IMG:
                plt.savefig(f'save_files/image{i}.png')
            plt.pause(0.0001)
            time.sleep(3)
            plt.close('all')



def generate_grid_data(path, N=30, M=30, num_env=10000, traj_per_env=5, Pmove_succ=1, Pobs_succ=1, world=1):
    """
    :param path: path for data file. use separate folders for training and test data
    :param N: grid rows
    :param M: grid columnts
    :param num_env: number of environments in the dataset (grids)
    :param traj_per_env: number of trajectories per environment (different initial state, goal, initial belief)
    :param Pmove_succ: probability of transition succeeding, otherwise stays in place
    :param Pobs_succ: probability of correct observation, independent in each direction
    """
        # ATATION - DEPPL LEARNING
    params = dotdict({
        'grid_n': N,
        'grid_m': M,
        'Pobst': 0.25,  # probability of obstacles in random grid

        'R_obst': -10, 'R_goal': 20, 'R_step': -0.1,
        'discount': 0.99,
        'Pmove_succ':Pmove_succ,
        'Pobs_succ': Pobs_succ,

        'num_action': 5,
        'moves': [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]],  # right, down, left, up, stay
        'stayaction': 4,

        'num_obs': 16,
        'observe_directions': [[0, 1], [1, 0], [0, -1], [-1, 0]],
        })

    params['obs_len'] = len(params['observe_directions'])
    params['num_state'] = params['grid_n']*params['grid_m']
    params['traj_limit'] = 4 * (params['grid_n'] + params['grid_m'])
    params['R_step'] = [params['R_step']] * params['num_action']
    # save params
    if not os.path.isdir(path):
        os.mkdir(path)
    pickle.dump(dict(params), open(path + "/params.pickle", 'wb'), -1)

    # randomize seeds, set to previous value to determinize random numbers
    np.random.seed()
    random.seed()

    # grid domain object
    domain = Grid(params, world=world)

    # make database file
    db = Grid.create_db(path+".example", params, num_env, traj_per_env)

    for env_i in range(num_env):
        print ("Generating env %d with %d trajectories "%(env_i, traj_per_env))
        domain.generate_example(db)

    print ("Done.")


def run_qmdp_net():
    arglist = ['./data/grid10/', '--loadmodel', './data/grid10/trained-model/final.chk',
               '--epochs', '0', '--lim_traj_len', '100']
    params = parse_args(arglist)
    params['grid_m'] = 20
    params['grid_n'] = 20
    params['Pmove_succ'] = 0.9
    params['Pobs_succ'] = 0.9
    params['K'] = 450
    modelfile = params.loadmodel[0]
    run_eval(params, modelfile,run_experiment=True)

if __name__ == '__main__':

        # qmdp
        generate_grid_data('grid_4_world_pmove', N=20, M=20, num_env=1, traj_per_env=10,
                       world=4,Pmove_succ=0.9, Pobs_succ=0.9)

        # qmdp-net
        run_qmdp_net()


        # gmdp with 10
        # generate_grid_data('grid_10_world_0_0_0_0', N=10, M=10, num_env=1, traj_per_env=10,
        #            Pmove_succ= 0.7 ,world=1)


        # generate_grid_data('grid_20_20', N=20, M=20, num_env=1, traj_per_env=10,
        #              Pmove_succ=1, Pobs_succ=1,world=4)






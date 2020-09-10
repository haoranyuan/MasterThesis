#from sklearn.svm import SVR
from Another_solver import ProjectionMethod
from rl_3D import RL_Multi
from rewardconstruct import FeatureExpectation, Rewardconstruct
from demo_discrete import Discretize
import os
import shutil
import numpy as np


DIR = 'AL_results/projection_test(3)/'
EP_LEN = 200
SHAPE = [21, 11, 2]
dis = Discretize()
# prior 2(693): 10, 3
MAX_NUM = 6
MIN_NUM = 6

def control_measure(traj):
    crash = 0
    for i in range(len(traj)):
        j = min(len(traj)-1, i+1)
        if traj[j][0] - traj[i][0] > 1:
            crash += 1
    score = -np.sum(np.absolute(traj[:, 0] - (SHAPE[0]-1)/2)) - 0 * crash
    return score


def sort_key(elem):
    # sort according to the first element
    return elem[0]


def list_traj(raw_data, discretize_flag=True, header_exist=True):
    # make a dictionary that contains score and states in each trajectory
    dis.data = raw_data
    if discretize_flag:
        dis_data = dis.discretize_data()
    else:
        dis_data = raw_data
    if header_exist:
        dis_traj = np.reshape(a=dis_data, newshape=[-1, EP_LEN+1, raw_data.shape[1]])
    else:
        dis_traj = np.reshape(a=dis_data, newshape=[-1, EP_LEN, raw_data.shape[1]])
    ranked_data = {}
    del_cont = 0
    for i in range(SHAPE[0]*SHAPE[1]):
        name = str(list(np.unravel_index(i, SHAPE[:2])))
        if name not in ranked_data.keys():
            ranked_data[name] = []
        for traj in dis_traj:
            if tuple(traj[0][0:2]) == np.unravel_index(i, SHAPE[:2]):
                if header_exist:
                    score = control_measure(traj[1:])
                else:
                    score = control_measure(traj)
                flat_traj = np.reshape(traj, [-1, ])
                ranked_data[name].append(np.concatenate((np.array([score]), flat_traj)))
        if ranked_data[name] == []:
            del ranked_data[name]
            del_cont += 1
            continue
        ranked_data[name].sort(reverse=True, key=sort_key)
        ranked_data[name] = ranked_data[name][:MAX_NUM]
    if del_cont != 0:
        print('deleted {} empty dicts.'.format(del_cont))
    return ranked_data


def trajectory_rerank(ranked_traj1, ranked_traj2):
    # replace the element with low score in ranked_traj1 with the element with high score in ranked_traj2
    for key in ranked_traj2:
        if key in ranked_traj1.keys():
            scores1 = set([sort_key(e) for e in ranked_traj1[key]])
            split_point = MAX_NUM
            ranked_traj1[key].extend(ranked_traj2[key])
            ranked_traj1[key].sort(reverse=True, key=sort_key)
            ranked_traj1[key] = ranked_traj1[key][:split_point]
            scores2 = set([sort_key(e) for e in ranked_traj1[key]])
            if len(scores2-scores1) > 0:
                print('filled and replaced {} of trajs in {}'.format(len(scores2-scores1), key))
        else:
            split_point = MIN_NUM
            if split_point == 0:
                continue
            ranked_traj1[key] = ranked_traj2[key]
            ranked_traj1[key].sort(reverse=True, key=sort_key)
            ranked_traj1[key] = ranked_traj1[key][:split_point]
            print('filled all trajs in {}'.format(key))

    return ranked_traj1


def reshape_trajdict(ranked_traj):
    # reshape the trajectory dictionary into a 2-dim numpy array
    for key in ranked_traj:
        for flat_traj_ in ranked_traj[key]:
            flat_traj = flat_traj_[1:]
            traj = np.reshape(flat_traj, newshape=[-1, len(SHAPE)])
            try:
                trajectory = np.vstack((trajectory, traj))
            except Exception:
                trajectory = traj
    return trajectory


def reward_reconstruct():
    FEA_EXP = FeatureExpectation()
    R = Rewardconstruct()

    if os.path.isfile(DIR + 'feature_expectations.csv'):
        # make a dictionary containing score of each trajectory of the demonstration
        # Also, don't discretize the state if it is already discrete
        demo_traj = np.genfromtxt(DIR + 'demodata.csv', delimiter=',')
        demo_dict = list_traj(raw_data=demo_traj, discretize_flag=False, header_exist=True)
        f_e = np.genfromtxt(DIR + 'feature_expectations.csv', delimiter=',')
        demo_history = np.genfromtxt(DIR + 'demo_history.csv', delimiter=',')
    else:
        # Also make a dictionary here
        demo_traj = np.genfromtxt(DIR + 'demodata.csv', delimiter=',')[:, :-1]
        demo_dict = list_traj(raw_data=demo_traj, discretize_flag=True, header_exist=True)
        dis.data = demo_traj
        demo_traj_s = dis.discretize_data()
        demo_feature_exp, _ = FEA_EXP.featureexpectations(trajectories=demo_traj_s, header_exist=True)
        demo_feature_exp = np.hstack((demo_feature_exp, np.array([1])))
        f_e = demo_feature_exp
        demo_history = demo_feature_exp[:-1]
    if f_e.ndim == 1:
        f_e = f_e[np.newaxis, :]  # Make sure feature expectation matrix has 2 dim
    if demo_history.ndim == 1:
        demo_history = demo_history[np.newaxis, :]

    iter_count = len(f_e) - 1
    agent_file_pre_iter = DIR+'iter'+str(iter_count-1)+'/'+'agentdata_AL.csv'
    agent_train_pre_iter = DIR+'iter'+str(iter_count-1)+'/'+'agenttrain_AL.csv'
    try:
        agent_file_path = shutil.copy(agent_file_pre_iter, DIR)
        agent_train_path = shutil.copy(agent_train_pre_iter, DIR)
        print('AL: copying agent trajectories of the previous step successfully at:', agent_file_pre_iter)
    except Exception:
        agent_file_path = DIR + 'agentdata_AL.csv'
        agent_train_path = DIR + 'agenttrain_AL.csv'
        print('AL: Fail to copy agent trajectories of the previous step')

    if os.path.isfile(agent_file_path):
        agent_traj = np.genfromtxt(agent_file_path, delimiter=',')
        agent_train_traj = np.genfromtxt(agent_train_path, delimiter=',')
        # make a dictionary containing the score of each trajectory
        agent_dict = list_traj(raw_data=np.vstack((agent_traj, agent_train_traj)), header_exist=True)

        # proceed to Apprenticeship learning routine
        dis.data = agent_traj
        agent_traj = dis.discretize_data()
        agent_feature_exp, _ = FEA_EXP.featureexpectations(trajectories=agent_traj, header_exist=True)
        print('AL: loading agent trajectories at iteration ', iter_count)
        demo_dict_reranked = trajectory_rerank(ranked_traj1=demo_dict, ranked_traj2=agent_dict)
        demo_traj_reranked = reshape_trajdict(demo_dict_reranked)
    else:
        # randomly initialise the agent feature expectations
        #np.random.seed(2)
        random_traj = np.genfromtxt('AL_results/other/agentdata_random.csv', delimiter=',')
        #agent_dict = list_traj(raw_data=random_traj, header_exist=True)
        dis.data = random_traj
        agent_traj = dis.discretize_data()
        agent_feature_exp, _ = FEA_EXP.featureexpectations(trajectories=agent_traj, header_exist=True)
        print('AL: randomly initialise agent policy')
        demo_traj_reranked = reshape_trajdict(demo_dict)

    demo_feature_exp_reranked, _ = FEA_EXP.featureexpectations(trajectories=demo_traj_reranked, header_exist=True)

    # Update feature expectations
    f_e[0] = np.concatenate((demo_feature_exp_reranked, np.array([1])))
    new_entry = np.concatenate((agent_feature_exp, np.array([-1])))
    f_e = np.vstack((f_e, new_entry))
    demo_history = np.vstack((demo_history, f_e[0, :-1]))

    train = np.array(f_e)
    omega, t = ProjectionMethod(feat_exp=train, dir=DIR)
    # compute the new reward
    reward_scheme = R.reward_scheme(omega, scale=1)
    reward_scheme = reward_scheme.reshape([462, ])
    # relocate the maximum reward to 0, and the minimum to -1
    reward_scheme = (reward_scheme - max(reward_scheme))/(max(reward_scheme) - min(reward_scheme))
    if not os.path.exists(DIR+'iter'+str(iter_count)+'/'):
        os.makedirs(DIR+'iter'+str(iter_count)+'/')  # Make directory for new iteration
        print('AL: make new directory for the new iteration at:', DIR+'iter'+str(iter_count)+'/')
    np.savetxt(DIR+'demodata.csv', demo_traj_reranked, delimiter=',')
    np.savetxt(DIR+'iter'+str(iter_count)+'/'+'reward_AL.csv', reward_scheme, delimiter=',')
    np.savetxt(DIR+'feature_expectations.csv', f_e, delimiter=',')
    np.savetxt(DIR+'demo_history.csv', demo_history, delimiter=',')

    return iter_count

if __name__ == "__main__":
    for _ in range(20):
        iter = reward_reconstruct()
        print('AL: iteration:', iter)
        if os.path.isfile(DIR+'iter'+str(iter)+'/'+'reward_AL.csv'):
            print('AL: reward saved at: ',DIR+'iter'+str(iter)+'/')
        else:
            input('reward not found')
        RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, reward_flag='AL', validation=0, render=False, episode=1000, file_dir=DIR+'iter'+str(iter)+'/', training_data=True)

        RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, reward_flag='AL', validation=1, render=False, episode=693, file_dir=DIR+'iter'+str(iter)+'/', fuzzy_validation=True)
        if os.path.isfile(DIR+'iter'+str(iter)+'/'+'agentdata_AL.csv'):
            print('AL: agent traj saved at:', DIR+'iter'+str(iter)+'/')
        else:
            input('agent traj not found')
    # ------------------------------------------
    print('AL: now showing the difference between experts feature expectations and the agents')
    fea = np.genfromtxt(DIR+'feature_expectations.csv', delimiter=',')[:, :-1]
    print([np.linalg.norm(fea_-fea[0]) for fea_ in fea[1:]])

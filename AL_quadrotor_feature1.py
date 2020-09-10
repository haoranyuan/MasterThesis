#from sklearn.svm import SVR
from Another_solver import ProjectionMethod
from rl_3D import RL_Multi
from rewardconstruct import FeatureExpectation, Rewardconstruct
from demo_discrete import Discretize
import os
import shutil
import numpy as np


DIR = 'AL_results/projection_test/'

SHAPE = [21, 11, 2]
NEW_SHAPE = [21, 5]

def refine_features(feature_expectations, cate):
    feature_expectations = feature_expectations.reshape(SHAPE)
    new = feature_expectations
    new_exp = np.hstack((new[:, 0], new[:, int((SHAPE[1]-1)/2)], new[:, -1]))
    return np.hstack((np.reshape(new_exp, newshape=[-1]), np.array([cate])))

def expand_features(omega):
    omega = np.reshape(omega, newshape=[21, 3, 2])
    new = np.repeat(omega, repeats=[5, 1, 5], axis=1)
    new_omega = np.empty(SHAPE)
    new_omega[:, :, 0] = np.copy(new)
    new_omega[:, :, 1] = np.copy(new)
    return new_omega.reshape([-1])

def reward_reconstruct():
    FEA_EXP = FeatureExpectation()
    R = Rewardconstruct()

    demo_traj = np.genfromtxt(DIR+'demodata.csv', delimiter=',')[:, :-1]
    dis = Discretize(data=demo_traj)
    if os.path.isfile(DIR + 'feature_expectations.csv'):
        # make a dictionary containing score of each trajectory of the demonstration
        # Also, don't discretize the state if it is already discrete
        f_e = np.genfromtxt(DIR + 'feature_expectations.csv', delimiter=',')
    else:
        # Also make a dictionary here
        demo_traj = np.genfromtxt(DIR + 'demodata.csv', delimiter=',')[:, :-1]
        dis.data = demo_traj
        demo_traj_s = dis.discretize_data()
        demo_feature_exp, _ = FEA_EXP.featureexpectations(trajectories=demo_traj_s, header_exist=True)
        # demo_feature_exp = np.hstack((demo_feature_exp, np.array([1])))
        demo_feature_exp = refine_features(demo_feature_exp, cate=1)
        f_e = demo_feature_exp
    if f_e.ndim == 1:
        f_e = f_e[np.newaxis, :]  # Make sure feature expectation matrix has 2 dim

    iter_count = len(f_e) - 1
    agent_file_pre_iter = DIR+'iter'+str(iter_count-1)+'/'+'agentdata_AL.csv'
    try:
        agent_file_path = shutil.copy(agent_file_pre_iter, DIR)
        print('AL: copying agent trajectories of the previous step successfully at:', agent_file_pre_iter)
    except Exception:
        agent_file_path = DIR + 'agentdata_AL.csv'
        print('AL: Fail to copy agent trajectories of the previous step')

    if os.path.isfile(agent_file_path):
        agent_traj = np.genfromtxt(agent_file_path, delimiter=',')
        dis.data = agent_traj
        agent_traj = dis.discretize_data()
        agent_feature_exp, _ = FEA_EXP.featureexpectations(trajectories=agent_traj, header_exist=True)
        print('AL: loading agent trajectories at iteration ', iter_count)
    else:
        # randomly initialise the agent feature expectations
        #np.random.seed(2)
        #rand_policy = np.random.random([462, ])
        #agent_feature_exp = rand_policy / sum(rand_policy) * 100
        random_traj = np.genfromtxt('AL_results/other/agentdata_random.csv', delimiter=',')
        dis.data = random_traj
        agent_traj = dis.discretize_data()
        agent_feature_exp, _ = FEA_EXP.featureexpectations(trajectories=agent_traj, header_exist=True)
        print('AL: randomly initialise agent policy')
    # Update feature expectations
    # new_entry = np.concatenate((agent_feature_exp, np.array([-1])))
    new_entry = refine_features(agent_feature_exp, cate=-1)
    f_e = np.vstack((f_e, new_entry))

    train = np.array(f_e)
    # Uncomment the following codes if want to use SVM instead of projection method
    '''
    linclf = SVR(kernel='linear')
    linclf.fit(train[:, :-1], train[:, -1])
    support_index = linclf.support_
    print(support_index)
    omega = np.squeeze(linclf.coef_)
    omega = omega / np.linalg.norm(omega)
    dif = np.transpose(np.array(f_e) - np.array(f_e[0]))

    dif = dif[0:-1]
    t = np.dot(omega, dif)
    '''
    omega, t = ProjectionMethod(feat_exp=train, dir=DIR)
    omega = expand_features(omega)
    #print('distance to the expert expectation:', t)
    reward_scheme = R.reward_scheme(omega, scale=1)
    reward_scheme = reward_scheme.reshape([462, ])
    # relocate the maximum reward to 0, and the minimum to -1
    reward_scheme = (reward_scheme - max(reward_scheme))/(max(reward_scheme) - min(reward_scheme))
    if not os.path.exists(DIR+'iter'+str(iter_count)+'/'):
        os.makedirs(DIR+'iter'+str(iter_count)+'/')  # Make directory for new iteration
        print('AL: make new directory for the new iteration at:', DIR+'iter'+str(iter_count)+'/')
    np.savetxt(DIR+'iter'+str(iter_count)+'/'+'reward_AL.csv', reward_scheme, delimiter=',')
    # input('AL: please check the reward function and then proceed')
    np.savetxt(DIR+'feature_expectations.csv', f_e, delimiter=',')
    return iter_count

if __name__ == "__main__":
    for _ in range(20):
        iter = reward_reconstruct()
        print('AL: iteration:', iter)
        if os.path.isfile(DIR+'iter'+str(iter)+'/'+'reward_AL.csv'):
            print('AL: reward saved at: ',DIR+'iter'+str(iter)+'/')
        else:
            input('reward not found')
        RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, reward_flag='AL', validation=0, render=False, episode=1000, file_dir=DIR+'iter'+str(iter)+'/')

        RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, reward_flag='AL', validation=1, render=False, episode=693, file_dir=DIR+'iter'+str(iter)+'/')
        if os.path.isfile(DIR+'iter'+str(iter)+'/'+'agentdata_AL.csv'):
            print('AL: agent traj saved at:', DIR+'iter'+str(iter)+'/')
        else:
            input('agent traj not found')
    # ------------------------------------------
    print('AL: now showing the difference between experts feature expectations and the agents')
    fea = np.genfromtxt(DIR+'feature_expectations.csv', delimiter=',')[:, :-1]
    print([np.linalg.norm(fea_-fea[0]) for fea_ in fea[1:]])

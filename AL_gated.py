#from sklearn.svm import SVR
from Another_solver import ProjectionMethod
from rl_3D import RL_Multi
from rewardconstruct import FeatureExpectation, Rewardconstruct
from demo_discrete import Discretize
import os
import shutil
import numpy as np


DIR = 'AL_results/gated_projection_2(693)/'

def control_measure(feature_exp, ep_len = 200):
    feature_exp = np.reshape(feature_exp, newshape=[21, 11, 2])
    target_feature = (10, 5)
    score = sum(feature_exp[target_feature])
    return score

def reward_reconstruct():
    FEA_EXP = FeatureExpectation()
    R = Rewardconstruct()

    demo_traj = np.genfromtxt(DIR+'demodata.csv', delimiter=',')[:, :-1]
    dis = Discretize(data=demo_traj)

    if os.path.isfile(DIR + 'feature_expectations.csv'):
        f_e = np.genfromtxt(DIR + 'feature_expectations.csv', delimiter=',')
        demo_history = np.genfromtxt(DIR + 'demo_history.csv', delimiter=',')
    else:
        demo_traj_s = dis.discretize_data()
        demo_feature_exp, _ = FEA_EXP.featureexpectations(trajectories=demo_traj_s)
        demo_feature_exp = np.hstack((demo_feature_exp, np.array([1])))
        f_e = demo_feature_exp
        demo_history = demo_feature_exp[:-1]
    if f_e.ndim == 1:
        f_e = f_e[np.newaxis, :]  # Make sure feature expectation matrix has 2 dim
    if demo_history.ndim == 1:
        demo_history = demo_history[np.newaxis, :]

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
        agent_feature_exp, _ = FEA_EXP.featureexpectations(trajectories=agent_traj)
        print('AL: loading agent trajectories at iteration ', iter_count)
        # Measure the control performance of the demonstration and the agent
        demo_score = control_measure(feature_exp=f_e[0, :-1])
        agent_score = control_measure(feature_exp=agent_feature_exp)
        print(demo_score, agent_score)
        if agent_score > demo_score:
            print('replace the demo with the agent data as agent performs better')
            f_e[0, :-1] = 0.5*agent_feature_exp + 0.5*f_e[0, :-1]


    else:
        # randomly initialise the agent feature expectations
        #np.random.seed(2)
        #rand_policy = np.random.random([462, ])
        #agent_feature_exp = rand_policy / sum(rand_policy) * 100
        random_traj = np.genfromtxt('AL_results/other/agentdata_random.csv', delimiter=',')
        dis.data = random_traj
        agent_traj = dis.discretize_data()
        agent_feature_exp, _ = FEA_EXP.featureexpectations(trajectories=agent_traj)
        print('AL: randomly initialise agent policy')
    # Update feature expectations
    new_entry = np.concatenate((agent_feature_exp, np.array([-1])))
    f_e = np.vstack((f_e, new_entry))
    demo_history = np.vstack((demo_history, f_e[0, :-1]))
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

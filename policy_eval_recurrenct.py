from matplotlib import pyplot as plt
import numpy as np
from rl_policy_dir import RL_Multi
from rewardconstruct import FeatureExpectation
from demo_discrete import Discretize
import os

def policy_eval(DIR, iter, PLOT=False, force_evaluate=False):
    # import mixed policies
    dis = Discretize(data=None)
    fe = FeatureExpectation()
    Feature_expectation = []
    # if not os.path.isfile(DIR + 'mixed_feature_expectations.csv') and force_evaluate:
    if True:
        print('Evaluating un-fuzzy policy')
        for i in range(iter):
            RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=False, validation=1,
                     episode=231, file_dir=DIR + 'iter' + str(i) + '/',
                     policy_dir=DIR + 'iter' + str(i) + '/mixed_policy.csv')
            dis.data = np.genfromtxt(DIR + 'iter' + str(i) + '/agentdata_mp.csv', delimiter=',')
            traj = dis.discretize_data()
            feat, _ = fe.featureexpectations(trajectories=traj, header_exist=True)
            Feature_expectation.append(feat)

        np.savetxt(DIR + 'mixed_feature_expectations.csv', Feature_expectation, delimiter=',')
    Feature_expectation = []
    # if not os.path.isfile(DIR + 'mixedfuzzy_feature_expectations.csv') and force_evaluate:
    if True:
        print('Evaluating fuzzy policy')
        for i in range(iter):
            RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=False, validation=1,
                     episode=231, file_dir=DIR + 'iter' + str(i) + '/',
                     policy_dir=DIR + 'iter' + str(i) + '/mixedfuzzy_policy.csv', fuzzy=True)
            dis.data = np.genfromtxt(DIR + 'iter' + str(i) + '/agentdata_fuzzymp.csv', delimiter=',')
            traj = dis.discretize_data()
            feat, _ = fe.featureexpectations(trajectories=traj, header_exist=True)
            Feature_expectation.append(feat)

        np.savetxt(DIR + 'mixedfuzzy_feature_expectations.csv', Feature_expectation, delimiter=',')


    mixed_fea = np.genfromtxt(DIR + 'mixed_feature_expectations.csv', delimiter=',')
    mixedfuzzy_fea = np.genfromtxt(DIR + 'mixedfuzzy_feature_expectations.csv', delimiter=',')
    fea__ = np.genfromtxt(DIR + 'feature_expectation_AL.csv', delimiter=',')
    fea_e = np.genfromtxt(DIR + 'feature_expectations.csv', delimiter=',')[0, :-1]
    fea = np.genfromtxt(DIR + 'feature_expectations.csv', delimiter=',')[1:, :-1]
    print([np.linalg.norm(f - fea_e) for f in fea])
    print([np.linalg.norm(f - fea_e) for f in fea__])
    print([np.linalg.norm(f - fea_e) for f in mixed_fea])
    print([np.linalg.norm(f - fea_e) for f in mixedfuzzy_fea])
    Lambda = np.genfromtxt(DIR + 'Lambda.csv', delimiter=',')
    '''
    #print([l for l in Lambda])
    lam = [l for l in Lambda]


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('distance to $\mu_e$')
    ax1.set_xlabel('iterations')
    ax1.plot([np.linalg.norm(f - fea_e) for f in fea], label='$\mu$ using reward functions')
    ax1.plot([np.linalg.norm(f - fea_e) for f in fea__], label=r"$\bar \mu$")
    ax1.plot([np.linalg.norm(f - fea_e) for f in mixedfuzzy_fea], label= 'mixed non-deterministic'+r'$\hat\mu$')
    ax1.plot([np.linalg.norm(f - fea_e) for f in mixed_fea], label= 'mixed deterministic'+r'$\hat\mu$')

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(lam))+1, lam, 'o--', label='$\lambda$')
    ax2.set_ylabel('$\lambda$')
    ax1.set_xticks(np.arange(0, 20))
    #ax2.set_yticks(lam)
    ax1.grid()
    fig.legend()
    '''
    if PLOT:
        plt.show()

if __name__ == '__main__':
    DIR = 'AL_results/projection_good/projection_good6/'
    iter = 20
    policy_eval(DIR, iter, PLOT=True)
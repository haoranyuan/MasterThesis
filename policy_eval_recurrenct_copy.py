from matplotlib import pyplot as plt
import numpy as np
from rl_policy_dir import RL_Multi
from rewardconstruct import FeatureExpectation
from demo_discrete import Discretize
import os

DIR = 'AL_results/p_projection_good_09noise/p_projection_good_09noise0/'
iter = 20
EXTEND = False

# import mixed policies
dis = Discretize(data=None)
fe = FeatureExpectation()
Feature_expectation = []
if not os.path.isfile(DIR + 'mixed_feature_expectations.csv'):
    print('Evaluating un-fuzzy policy')
    for i in range(iter):
        RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=False, validation=1,
                 episode=693, file_dir=DIR + 'iter' + str(i) + '/',
                 policy_dir=DIR + 'iter' + str(i) + '/mixed_policy.csv')
        dis.data = np.genfromtxt(DIR + 'iter' + str(i) + '/agentdata_mp.csv', delimiter=',')
        traj = dis.discretize_data()
        feat, _ = fe.featureexpectations(trajectories=traj, header_exist=True)
        Feature_expectation.append(feat)
    np.savetxt(DIR + 'mixed_feature_expectations.csv', Feature_expectation, delimiter=',')
else:
    print('extending feature expectation')
    feat_exp = np.genfromtxt(DIR + 'mixed_feature_expectations.csv', delimiter=',')
    if EXTEND:
        for i in range(iter):
            if not os.path.isfile(DIR+'iter' + str(i) + '/agentdata_mp.csv'):
                RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=False, validation=1,
                         episode=693, file_dir=DIR + 'iter' + str(i) + '/',
                         policy_dir=DIR + 'iter' + str(i) + '/mixed_policy.csv')
                dis.data = np.genfromtxt(DIR + 'iter' + str(i) + '/agentdata_mp.csv', delimiter=',')
                traj = dis.discretize_data()
                feat, _ = fe.featureexpectations(trajectories=traj, header_exist=True)
                feat_exp = np.vstack((feat_exp, feat))
        np.savetxt(DIR + 'mixed_feature_expectations.csv', feat_exp, delimiter=',')

Feature_expectation = []
if not os.path.isfile(DIR + 'mixedfuzzy_feature_expectations.csv'):
    print('Evaluating fuzzy policy')
    for i in range(iter):
        RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=False, validation=1,
                 episode=693, file_dir=DIR + 'iter' + str(i) + '/',
                 policy_dir=DIR + 'iter' + str(i) + '/mixedfuzzy_policy.csv', fuzzy=True)
        dis.data = np.genfromtxt(DIR + 'iter' + str(i) + '/agentdata_fuzzymp.csv', delimiter=',')
        traj = dis.discretize_data()
        feat, _ = fe.featureexpectations(trajectories=traj, header_exist=True)
        Feature_expectation.append(feat)

    np.savetxt(DIR + 'mixedfuzzy_feature_expectations.csv', Feature_expectation, delimiter=',')
else:
    print('extending feature expectation')
    feat_exp = np.genfromtxt(DIR + 'mixedfuzzy_feature_expectations.csv', delimiter=',')
    if EXTEND:
        for i in range(iter):
            if not os.path.isfile(DIR+'iter' + str(i) + '/agentdata_fuzzymp.csv'):
                RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=False, validation=1,
                         episode=693, file_dir=DIR + 'iter' + str(i) + '/',
                         policy_dir=DIR + 'iter' + str(i) + '/mixedfuzzy_policy.csv', fuzzy=True)
                dis.data = np.genfromtxt(DIR + 'iter' + str(i) + '/agentdata_fuzzymp.csv', delimiter=',')
                traj = dis.discretize_data()
                feat, _ = fe.featureexpectations(trajectories=traj, header_exist=True)
                feat_exp = np.vstack((feat_exp, feat))
        np.savetxt(DIR + 'mixedfuzzy_feature_expectations.csv', feat_exp, delimiter=',')

mixed_fea = np.genfromtxt(DIR + 'mixed_feature_expectations.csv', delimiter=',')
mixedfuzzy_fea = np.genfromtxt(DIR + 'mixedfuzzy_feature_expectations.csv', delimiter=',')
fea__ = np.genfromtxt(DIR + 'feature_expectation_AL.csv', delimiter=',')
fea_e = np.genfromtxt(DIR + 'feature_expectations.csv', delimiter=',')[0, :-1]
fea = np.genfromtxt(DIR + 'feature_expectations.csv', delimiter=',')[1:, :-1]
his_fea = np.genfromtxt(DIR + 'demo_history.csv', delimiter=',')[1:]
print([np.linalg.norm(f - fea_e) for f in fea])
print([np.linalg.norm(f - fea_e) for f in fea__])
print([np.linalg.norm(f - fea_e) for f in mixed_fea])
print([np.linalg.norm(f - fea_e) for f in mixedfuzzy_fea])
Lambda = np.genfromtxt(DIR + 'iter19/'+'Lambda.csv', delimiter=',')
print([l for l in Lambda])
lam = [l for l in Lambda]


fig = plt.figure(figsize=[10, 5])
ax1 = fig.add_subplot(111)
ax1.set_ylabel('distance to $\mu_e$')
ax1.set_xlabel('iterations')
ax1.plot([np.linalg.norm(f - his_fea[i]) for i, f in enumerate(fea)], label='$\mu$ using reward functions')
ax1.plot([np.linalg.norm(f - his_fea[i]) for i, f in enumerate(fea__)], label=r"$\bar \mu$")
ax1.plot([np.linalg.norm(f - his_fea[i]) for i, f in enumerate(mixedfuzzy_fea)], label= 'mixed non-deterministic '+r'$\hat\mu$')
ax1.plot([np.linalg.norm(f - his_fea[i]) for i, f in enumerate(mixed_fea)], label= 'mixed deterministic '+r'$\hat\mu$')
ax1.plot([np.linalg.norm(f - fea_e) for f in his_fea], label= 'feature expectations history demo '+r'$\hat\mu_E$')
ax2 = ax1.twinx()
ax2.plot(np.arange(len(lam))+1, lam, 'o--', label='$\lambda$')
ax2.set_ylabel('$\lambda$')
ax1.set_xticks(np.arange(0, iter))
#ax2.set_yticks(lam)
ax1.grid()
fig.legend()
plt.show()

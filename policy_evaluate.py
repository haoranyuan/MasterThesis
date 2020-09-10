from matplotlib import pyplot as plt
import numpy as np
from rl_policy_dir import RL_Multi
from rewardconstruct import FeatureExpectation
from demo_discrete import Discretize
import os
DIR = 'AL_results/projection_good/projection_good8/'
# import mixed policies
dis = Discretize(data=None)
fe = FeatureExpectation()

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
#print([l for l in Lambda])
lam = [l for l in Lambda]


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('$||\mu-\mu_D||_2$')
ax1.set_xlabel('iterations')
ax1.plot([np.linalg.norm(f - fea_e) for f in fea], label='AL-RL')
ax1.plot([np.linalg.norm(f - fea_e) for f in fea__], label=r"$\bar \mu$")
#ax1.plot([np.linalg.norm(f - fea_e) for f in mixedfuzzy_fea], label= 'mixed non-deterministic'+r'$\hat\mu$')
ax1.plot([np.linalg.norm(f - fea_e) for f in mixed_fea], label= 'AL-PFP')

ax2 = ax1.twinx()
ax2.plot(np.arange(len(lam))+1, lam, 'o--', label='$\lambda$')
ax2.set_ylabel('$\lambda$')
ax1.set_xticks(np.arange(0, 20))
#ax2.set_yticks(lam)
fig.legend(bbox_to_anchor=(0.9, 0.89))
ax1.grid(linestyle=':')
plt.show()
from rewardconstruct import FeatureExpectation, Rewardconstruct
from demo_discrete import Discretize
from matplotlib import pyplot as plt
import numpy as np
''''''
FEA_EXP = FeatureExpectation()

agent_traj = np.genfromtxt('demodata.csv', delimiter=',')[:, :-1]
dis = Discretize(data=agent_traj)
dis_agent = dis.discretize_data()
mu_a, _ = FEA_EXP.featureexpectations(trajectories=dis_agent)

#np.savetxt('AL_results/other/feature_expectation_PIDpolicy.csv', mu_a, delimiter=',')

#mu_a = np.genfromtxt('AL_results/other/feature_expectation_mixedpolicy.csv', delimiter=',')
mu = np.genfromtxt('AL_results/projection_1/feature_expectations.csv', delimiter=',')[:, :-1]
#mu_e = mu[0, :]
mu_e = np.genfromtxt('AL_results/projection_1/feature_expectations.csv', delimiter=',')[0, :-1]
mu = mu[1:, :]
fig, axs = plt.subplots(1, 2, figsize=(5, 5), sharex=True, sharey=True)
fea_agent = mu_a.reshape([21, 11, 2])
axs[0].imshow(fea_agent[:, :, 0])
axs[1].imshow(fea_agent[:, :, 1])

fig1, axs1 = plt.subplots(1, 2, figsize=(5, 5), sharex=True, sharey=True)
fea_agent = mu_e.reshape([21, 11, 2])
axs1[0].imshow(fea_agent[:, :, 0])
axs1[1].imshow(fea_agent[:, :, 1])

fig2, axs2 = plt.subplots(1, 2, figsize=(5, 5), sharex=True, sharey=True)
fea_agent = mu_e.reshape([21, 11, 2]) - mu_a.reshape([21, 11, 2])
axs2[0].imshow(fea_agent[:, :, 0])
axs2[1].imshow(fea_agent[:, :, 1])

print(np.linalg.norm(mu_e-mu_a))
print([np.linalg.norm(mu_e-m) for m in mu])

plt.show()
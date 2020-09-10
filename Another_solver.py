import numpy as np
'''
from rewardconstruct import Rewardconstruct
from matplotlib import pyplot as plt
from os.path import isfile
'''


def ProjectionMethod(feat_exp, dir):
    #feat_exp = np.genfromtxt('feature_expectations.csv', delimiter=',')
    feat_exp = feat_exp[:, :-1]

    # R = Rewardconstruct()
    mu_e = feat_exp[0, :]
    MU = feat_exp[1:, :]
    if len(MU) == 1:
        MU_ = np.copy(MU)
        np.savetxt(dir+'feature_expectation_AL.csv', MU_, delimiter=',')
    MU_ = np.genfromtxt(dir+'feature_expectation_AL.csv', delimiter=',')
    if MU_.ndim == 1:
        MU_ = MU_[np.newaxis, :]
    if len(MU) == 1:
        mu_ = MU_[-1, :]
    else:
        mu_ = MU_[-1, :] + (np.dot((MU[-1, :] - MU_[-1, :]), (mu_e - MU_[-1, :]))) / (np.dot((MU[-1, :] - MU_[-1, :]),
                                                                                          (MU[-1, :] - MU_[-1, :]))) \
                                                                                            * (MU[-1, :] - MU_[-1, :])
        MU_ = np.vstack((MU_, mu_))
    np.savetxt(dir+'feature_expectation_AL.csv', MU_, delimiter=',')

    omega = mu_e - mu_
    t = np.linalg.norm(mu_e - mu_)
    print([np.linalg.norm(fe) for fe in MU_-mu_e])
    return omega, t



'''
r = R.reward_scheme(omega=w, scale=1)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2)
im = ax.imshow(r[:, :, 0])
im1 = ax1.imshow(r[:, :, 1])

plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np

DIR = 'AL_results/projection_good/'
fig1, axs1 = plt.subplots(1, 3, figsize=(12, 5), facecolor='w', edgecolor='k', sharex=True, sharey=True)
fig1.tight_layout()
fig1.subplots_adjust(hspace=0.12, wspace=0)
fig1.text(0.04, 0.5, 'Position', va='center', rotation='vertical')
axs1 = axs1.ravel()

file = DIR + 'projection_good0/iter19/sav_AL.csv'
sav = np.genfromtxt(file, delimiter=',').reshape([21, 11, 2])
policy = np.zeros(shape=sav[:, :, 0].shape)
for ij, sa1 in enumerate(sav):
    for j, sa in enumerate(sa1):
        a = np.argmax(sa)
        policy[ij, j] = a
axs1[0].imshow(policy, cmap='Blues')
axs1[0].set_title('RL in AL')

file = DIR + 'projection_good0/iter19/mixed_policy.csv'
policy = np.genfromtxt(file, delimiter=',')
policy = np.reshape(policy, newshape=[21, 11])
axs1[1].imshow(policy, cmap='Blues')
axs1[1].set_title('PFP')

from test import PID_Multi
PID = PID_Multi()
for ij, sa1 in enumerate(sav):
    for j, _ in enumerate(sa1):
        state = np.array([ij, j])
        policy[ij, j] = PID(state=state)
axs1[2].imshow(policy, cmap='Blues')
axs1[2].set_title('PID')

plt.show()

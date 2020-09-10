import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window):
    weights = np.repeat(1, window)/window
    return np.convolve(data, weights, mode='valid')

''''''
DIR = 'AL_results/projection_good_07noise/projection_good_07noise1/feature_expectations.csv'
# DIR = 'AL_results/projection_good/projection_good0/feature_expectations.csv'
sav = np.genfromtxt(DIR, delimiter=',')[2, :-1]
sav = np.reshape(sav, [21, 11, 2])
sav_partial1 = sav[:, :, 0]
sav_partial2 = sav[:, :, 1]
fig1 = plt.figure(num=3)
ax10 = fig1.add_subplot(1, 2, 1)
ax11 = fig1.add_subplot(1, 2, 2)
im10 = ax10.imshow(sav_partial1)
im11 = ax11.imshow(sav_partial2)


'''
fig, axs = plt.subplots(5, 4, figsize=(5, 12), facecolor='w', edgecolor='k', sharex=True, sharey=True)
fig.tight_layout()
fig.subplots_adjust(hspace=0.12, wspace=0)
#fig.text(0.5, 0.04, 'Speed', ha='center')
fig.text(0.04, 0.5, 'Position', va='center', rotation='vertical')
axs = axs.ravel()

for i in range(9):
    file = 'AL_results/projection_500training/iter' + str(i) + '/sav_AL.csv'
    sav = np.genfromtxt(file, delimiter=',').reshape([21, 11, 2])
    policy = np.zeros(shape=sav[:, :, 0].shape)
    for ij, sa1 in enumerate(sav):
        for j, sa in enumerate(sa1):
            a = np.argmax(sa)
            policy[len(policy[:, 0]) - ij - 1, j] = a
    axs[i].imshow(policy, cmap='tab20c')
    axs[i].set_title('iter '+str(i), fontsize=8)
plt.setp(axs, xticks=[0, 5, 10], xticklabels=['-0.5', '0', '+0.5'],
        yticks=[0, 10, 20], yticklabels=['+3', '0', '-3'])
#axs.xticks(np.arange(-0.3, 0.3, 0.3))
'''
'''
fig1, axs1 = plt.subplots(1, 3, figsize=(12, 5), facecolor='w', edgecolor='k', sharex=True, sharey=True)
fig1.tight_layout()
fig1.subplots_adjust(hspace=0.12, wspace=0)
fig1.text(0.04, 0.5, 'Position', va='center', rotation='vertical')
axs1 = axs1.ravel()

file = 'AL_results/RL_3000ep_per_iter/sav_A.csv'
sav = np.genfromtxt(file, delimiter=',').reshape([21, 11, 2])
policy = np.zeros(shape=sav[:, :, 0].shape)
for ij, sa1 in enumerate(sav):
    for j, sa in enumerate(sa1):
        a = np.argmax(sa)
        policy[len(policy[:, 0]) - ij - 1, j] = a
axs1[0].imshow(policy, cmap='tab20c')
axs1[0].set_title('Handcrafted reward')

from test import PID_Multi
PID = PID_Multi()
for ij, sa1 in enumerate(sav):
    for j, _ in enumerate(sa1):
        state = np.array([ij, j])
        policy[len(policy[:, 0]) - ij - 1, j] = PID(state=state)
axs1[1].imshow(policy, cmap='tab20c')
axs1[1].set_title('PID')
file = 'sav_AL.csv'
sav = np.genfromtxt(file, delimiter=',').reshape([21, 11, 2])
policy = np.zeros(shape=sav[:, :, 0].shape)
for ij, sa1 in enumerate(sav):
    for j, sa in enumerate(sa1):
        a = np.argmax(sa)
        policy[len(policy[:, 0]) - ij - 1, j] = a
axs1[2].imshow(policy, cmap='tab20c')
axs1[2].set_title('Reward iter19')
axs1[1].grid()
axs1[0].grid()
axs1[2].grid()
plt.setp(axs1, xticks=[0, 5, 10], xticklabels=['-0.5', '0', '+0.5'],
        yticks=[0, 10, 20], yticklabels=['+3', '0', '-3'])
'''
'''
reward = np.genfromtxt(DIR+'iter9/reward_AL.csv', delimiter=',')
reward = np.reshape(reward, [21, 11, 2])

reward_partial1 = reward[:, :, 0]
reward_partial2 = reward[:, :, 1]
fig0 = plt.figure(num=2,figsize=(5, 5))
ax00 = fig0.add_subplot(1, 2, 1)
ax = fig0.add_subplot(1, 2, 2)
im00 = ax00.imshow(reward_partial1)
imm = ax.imshow(reward_partial2)
fig0.suptitle('reward function')
'''
'''

idn = 0
iter = 20
total_ep = 2000
success_scatter = np.empty(shape=[iter, total_ep])
for i in range(iter):
    file = 'AL_results/projection_good_init/iter'+str(i)+'/sc_scatter_ALval1.csv'
    success_scatter[i] = np.genfromtxt(file, delimiter=',')[:, idn]

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)

plt.title('Success Rate Comparison')
plt.xlabel('iteration')
plt.ylabel('success rate')
success_scatter1 = moving_average(np.genfromtxt('sc_scatter_defaultval0.csv', delimiter=',')[:, idn], 200)[:3000]
success_scatter2 = moving_average(np.genfromtxt('sc_scatter_ALval0.csv', delimiter=',')[:, idn], 200)[:3000]
success_scatter3 = moving_average(np.genfromtxt('sc_scatter_Yurival0.csv', delimiter=',')[:, idn], 200)[:3000]

ax1.plot(np.arange(0, len(success_scatter1)), np.asarray(success_scatter1), label='default reward')
ax1.plot(np.arange(0, len(success_scatter2)), np.asarray(success_scatter2), label='default reward')
ax1.plot(np.arange(0, len(success_scatter3)), np.asarray(success_scatter3), label='default reward')

for i in range(iter):
    plt_data = moving_average(success_scatter[i], window=200)
    ax1.plot(np.arange(0, len(plt_data)), plt_data, label='iter'+str(i))
#ax1.legend()
ax1.grid()

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
plt.title('Success Ratio')
plt.xlabel('iteration of AL')
plt.ylabel('success ratio')
plt.xticks(ticks=np.arange(0, 20, 1))
sc_average = np.empty(shape=iter)
for i in range(iter):
    sc_average[i] = sum(success_scatter[i])/total_ep
ax2.plot(np.arange(0, len(sc_average)), sc_average)
ax2.grid()
'''
plt.show()



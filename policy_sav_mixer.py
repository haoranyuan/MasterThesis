import numpy as np
from matplotlib import pyplot as plt
from test import PID_Multi

fig, axs = plt.subplots(1, 20, figsize=(15, 3), facecolor='w', edgecolor='k', sharex=True, sharey=True)
fig.tight_layout()
fig.subplots_adjust(hspace=0., wspace=0.2)
axs = axs.ravel()

for ii in range(21):
    if ii < 1:
        '''
        PID = PID_Multi()
        for ij in range(21):
            for j in range(11):
                state = np.array([ij, j])
                policy_mix[ij, j] = PID(state=state)
        '''
        continue
    else:
        #DIR = 'AL_results/projection_firsthalf_position/'
        DIR = 'AL_results/projection_1(693)/'
        DIR_1 = DIR+'feature_expectations.csv'
        DIR_2 = DIR+'feature_expectation_AL.csv'
        POLICY_COUNT = ii

        fea_exp = np.genfromtxt(DIR_1, delimiter=',')
        mu_e = fea_exp[0, :-1]
        MU = fea_exp[1:, :-1]

        MU_ = np.genfromtxt(DIR_2, delimiter=',')
        Lambda = np.ones(shape=(POLICY_COUNT-1, ))
        for i in np.arange(0, POLICY_COUNT):
            fe = MU[i, :]
            if i == 0:
                continue
            fe_ = MU_[i-1, :]
            #MU_ = fe_ + (np.dot((fe - fe_), (mu_e - fe_))) / (np.dot((fe - fe_), (fe - fe_))) * (fe - fe_)
            Lambda[i-1] = np.clip((np.dot((fe - fe_), (mu_e - fe_))) / (np.dot((fe - fe_), (fe - fe_))), 0, 1)
        #np.random.seed(1)
        #Lambda = np.random.random(size=Lambda.size)
        #Lambda[:int(len(Lambda)/2)] = np.absolute(np.sort(-Lambda[:int(len(Lambda)/2)]))
        MUL = 1
        for i in reversed(range(POLICY_COUNT)):
            if i-1 == -1:
                sav = -np.random.random(size=[21, 11, 2])
            else:
                file = DIR+'iter' + str(i-1) + '/sav_AL.csv'
                sav = np.genfromtxt(file, delimiter=',').reshape([21, 11, 2])

            if i == POLICY_COUNT - 1:
                sav_mix = np.zeros_like(sav)
            sav = sav/abs(min(sav.reshape([1, -1]).squeeze()))
            if i > 0:
                sav_mix += Lambda[i-1] * MUL * sav
                MUL *= 1-Lambda[i-1]
            else:
                sav_mix += MUL * sav
        sav_mix = sav_mix/abs(min(sav_mix.reshape([1, -1]).squeeze()))
        policy_mix = np.zeros(shape=sav[:, :, 0].shape)
        for ij, sa1 in enumerate(sav_mix):
            for j, sa in enumerate(sa1):
                a = np.argmax(sa)
                policy_mix[ij, j] = a
        policy_mix = np.clip(policy_mix, 0, 1)
        #np.savetxt(DIR+'iter'+str(ii-1)+'/mixedfuzzy_policy.csv', policy_mix, delimiter=',')
        policy_mix = np.around(policy_mix)
        axs[ii-1].imshow(policy_mix, cmap='Blues')
        #np.savetxt(DIR+'iter'+str(ii-1)+'/mixed_policy.csv', policy_mix, delimiter=',')
        #axs[ii].set_title('lambda={}'.format(Lambda[POLICY_COUNT]))
        # axs[ii].grid()

PID = PID_Multi()
policy_PID = np.empty(shape=(21, 11))
for ij in range(21):
    for j in range(11):
        state = np.array([ij, j])
        policy_PID[ij, j] = PID(state=state)
fig = plt.figure(figsize=[6, 6])
ax1 = fig.add_subplot(111)
ax1.imshow(policy_PID, cmap='Blues')
ax1.set_title('PID policy')

print(Lambda)
plt.show()

#np.savetxt('AL_results/other/policy_mixed.csv', policy_mix, delimiter=',')
#np.savetxt('AL_results/other/policy_PID.csv', policy, delimiter=',')
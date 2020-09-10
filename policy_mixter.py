import numpy as np
from matplotlib import pyplot as plt
from test import PID_Multi

#DIR = 'AL_results/prior_projection_bad(1buffer)/'

def policy_mix(DIR, PRIOR=False, EP_number=21, PLOT=False, savefile=True):
    fig, axs = plt.subplots(1, 20, figsize=(15, 3), facecolor='w', edgecolor='k', sharex=True, sharey=True)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0., wspace=0.2)
    axs = axs.ravel()
    for ii in range(EP_number):
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
            DIR_1 = DIR + 'feature_expectations.csv'
            DIR_2 = DIR + 'feature_expectation_AL.csv'
            DIR_3 = DIR + 'demo_history.csv'  # load feature expectations of demonstrations
            POLICY_COUNT = ii

            fea_exp = np.genfromtxt(DIR_1, delimiter=',')
            #mu_e = fea_exp[0, :-1]
            if PRIOR:
                MU_e = np.genfromtxt(DIR_3, delimiter=',')[1:]
                mu_e = MU_e[ii - 1]
            else:
                mu_e = fea_exp[0, :-1]
            MU = fea_exp[1:, :-1]
            #if ii == 30:
                #print(ii)
            MU_ = np.genfromtxt(DIR_2, delimiter=',')
            Lambda = np.ones(shape=(POLICY_COUNT-1, ))

            for i in np.arange(0, POLICY_COUNT):
                fe = MU[i, :]
                if i == 0:
                    continue
                fe_ = MU_[i-1, :]
                #MU_ = fe_ + (np.dot((fe - fe_), (mu_e - fe_))) / (np.dot((fe - fe_), (fe - fe_))) * (fe - fe_)
                #mu_e = MU_e[i]
                #mu_e = fea_exp[0, :-1]
                Lambda[i-1] = np.clip((np.dot((fe - fe_), (mu_e - fe_))) / (np.dot((fe - fe_), (fe - fe_))), 0, 1)
            MUL = 1
            for i in reversed(range(POLICY_COUNT)):
                if i-1 == -1:
                    policy = np.genfromtxt('AL_results/other/random_policy.csv', delimiter=',')
                else:
                    file = DIR+'iter' + str(i-1) + '/sav_AL.csv'
                    sav = np.genfromtxt(file, delimiter=',').reshape([21, 11, 2])
                    policy = np.zeros(shape=sav[:, :, 0].shape)
                    for ij, sa1 in enumerate(sav):
                        for j, sa in enumerate(sa1):
                            a = np.argmax(sa)
                            policy[ij, j] = a
                if i == POLICY_COUNT - 1:
                    policy_mix = np.zeros_like(policy)

                if i > 0:
                    policy_mix += Lambda[i-1] * MUL * policy
                    MUL *= 1-Lambda[i-1]
                else:
                    policy_mix += MUL * policy
            policy_mix = np.clip(policy_mix, 0, 1)
            #print(Lambda)
            if savefile:
                np.savetxt(DIR + 'iter'+str(ii-1)+'/Lambda.csv', Lambda, delimiter=',')
                np.savetxt(DIR+'iter'+str(ii-1)+'/mixedfuzzy_policy.csv', policy_mix, delimiter=',')
            policy_mix = np.around(policy_mix)
            axs[ii-1].imshow(policy_mix, cmap='Blues')
            axs[ii-1].set_title('iter '+str(ii-1))
            np.savetxt(DIR+'iter'+str(ii-1)+'/mixed_policy.csv', policy_mix, delimiter=',')
            #axs[ii].set_title('lambda={}'.format(Lambda[POLICY_COUNT]))
            # axs[ii].grid()
    if savefile:
        np.savetxt(DIR + 'Lambda.csv', Lambda, delimiter=',')
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
    if PLOT:
        plt.show()

#np.savetxt('AL_results/other/policy_mixed.csv', policy_mix, delimiter=',')
#np.savetxt('AL_results/other/policy_PID.csv', policy, delimiter=',')

if __name__ == '__main__':
    # DIR = 'AL_results/p_projection_good/p_projection_good0/'
    DIR = 'AL_results/p_projection_good_09noise/p_projection_good_09noise0/'
    # DIR = 'AL_results/p_projection_bad/p_projection_bad0/'
    policy_mix(DIR, PRIOR=True, EP_number=21, PLOT=True, savefile=False)
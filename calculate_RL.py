import numpy as np
import os
from matplotlib import pyplot as plt
from demo_discrete import Discretize
from tqdm import tqdm
ep_len = 201
Range = {
        'linear_z': [-3, 3],
        'linear_zrate': [-0.5, 0.5]}

DIR_ = 'RL_results/'
Dis = Discretize()
iter = 20
REPEATS = 10
for r in range(REPEATS):
    DIR = DIR_ + 'RL'+str(r)
    if not os.path.isfile(DIR+'/score_file.csv'):
            crash_count = np.empty(shape=[1, iter])
            average_error = np.empty(shape=[1, iter])
            for i in tqdm(range(iter)):
                agentdata = np.genfromtxt(DIR+'/agentdata_A'+str(i)+'.csv', delimiter=',')
                Dis.data = agentdata
                agentdata = Dis.discretize_data()
                agentdata_ = np.reshape(agentdata, newshape=[-1, ep_len, agentdata.shape[-1]])
                crash_number = 0
                total_e = 0
                for trial in agentdata_:
                    for n, state in enumerate(trial):
                        #if n>=100:
                        total_e += abs(state[0]-10)
                        if state[0] < Range['linear_z'][0] or state[0] > Range['linear_z'][1]:
                            crash_number += 1
                crash_count[0, i] = crash_number/agentdata_.shape[0]
                average_error[0, i] = total_e/agentdata_.shape[0]
            np.savetxt(DIR + '/score_file.csv', np.vstack((crash_count, average_error)), delimiter=',')

score = np.zeros(shape=[REPEATS, 2, iter])
for i in range(REPEATS):
    score[i, :, :] = np.genfromtxt(DIR_ + 'RL'+str(i) + '/score_file.csv', delimiter=',')

score_aver = np.average(score, axis=0)
score_error = np.std(score, axis=0)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('iterations')
ax2.set_ylabel('average tracking error')
_, caps, bars = ax2.errorbar(np.arange(0, iter), score_aver[1], capsize=3, yerr=score_error[1], label='SAL')
np.savetxt(DIR_ + '/RL_score.csv', np.vstack((score_aver, score_error)), delimiter=',')
plt.show()
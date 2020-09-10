import numpy as np
import os
from matplotlib import pyplot as plt
from demo_discrete import Discretize
from tqdm import tqdm

def control_eval(DIR, iter = 20, PLOT=False, force_evaluate=False):
    ep_len = 201
    Range = {
        'linear_z': [-3, 3],
        'linear_zrate': [-0.5, 0.5]}
    K1 = 0  # score coefficient for failure
    K2 = 1  # score coefficient for tracking error
    Dis = Discretize()

    #if not os.path.isfile(DIR+'score_file.csv') and force_evaluate:
    if True:

        crash_count = np.empty(shape=[3, iter])
        average_error = np.empty(shape=[3, iter])
        for i in tqdm(range(iter)):
            agentdata = np.genfromtxt(DIR+'iter'+str(i)+'/agentdata_AL.csv', delimiter=',')
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

        for i in tqdm(range(iter)):
            agentdata = np.genfromtxt(DIR + 'iter' + str(i) + '/agentdata_fuzzymp.csv', delimiter=',')
            Dis.data = agentdata
            agentdata = Dis.discretize_data()
            agentdata_ = np.reshape(agentdata, newshape=[-1, ep_len, agentdata.shape[-1]])
            crash_number = 0
            total_e = 0
            for trial in agentdata_:
                for state in trial:
                    total_e += abs(state[0]-10)
                    if state[0] < Range['linear_z'][0] or state[0] > Range['linear_z'][1]:
                        crash_number += 1
            crash_count[1, i] = crash_number/agentdata_.shape[0]
            average_error[1, i] = total_e / agentdata_.shape[0]

        for i in tqdm(range(iter)):
            agentdata = np.genfromtxt(DIR + 'iter' + str(i) + '/agentdata_mp.csv', delimiter=',')
            Dis.data = agentdata
            agentdata = Dis.discretize_data()
            agentdata_ = np.reshape(agentdata, newshape=[-1, ep_len, agentdata.shape[-1]])
            crash_number = 0
            total_e = 0
            for trial in agentdata_:
                for state in trial:
                    total_e += abs(state[0]-10)
                    if state[0] < Range['linear_z'][0] or state[0] > Range['linear_z'][1]:
                        crash_number += 1
            crash_count[2, i] = crash_number/agentdata_.shape[0]
            average_error[2, i] = total_e / agentdata_.shape[0]

        np.savetxt(DIR+'score_file.csv', np.vstack((crash_count, average_error)), delimiter=',')

    score_matrix = np.genfromtxt(DIR+'score_file.csv', delimiter=',')
    score = -K1*score_matrix[0:3] - K2 * score_matrix[3:]

    #demodata = np.genfromtxt('AL_results/Demonstrations/' + 'demodata_bad_headers.csv', delimiter=',')[:, :-1]
    demodata = np.genfromtxt('RL_results/agentdata_A0.csv', delimiter=',')
    Dis.data = demodata
    demodata = Dis.discretize_data()
    demodata_ = np.reshape(demodata, newshape=[-1, ep_len, demodata.shape[-1]])
    crash_number = 0
    total_e = 0
    for trial in demodata_:
        for state in trial:
            total_e += abs(state[0]-10)
            if state[0] < Range['linear_z'][0] or state[0] > Range['linear_z'][1]:
                    crash_number += 1
    crash_count = crash_number/demodata_.shape[0]
    average_error = total_e / demodata_.shape[0]

    demo_score = -K1*crash_count - K2 * average_error

    iteration = 30
    crash_count = np.empty(shape=(iteration, ))
    average_error = np.empty(shape=(iteration, ))
    agent_file_name = DIR + 'score_file_agent.csv'
    '''
    if not os.path.isfile(agent_file_name):
        for i in tqdm(range(iteration)):
            demodata = np.genfromtxt('AL_results/RL_3000ep_per_iter/agentdata_A'+str(i)+'.csv', delimiter=',')
            Dis.data = demodata
            demodata = Dis.discretize_data()
            demodata_ = np.reshape(demodata, newshape=[-1, ep_len, demodata.shape[-1]])
            crash_number = 0
            total_e = 0
            for trial in demodata_:
                for state in trial:
                    total_e += abs(state[0]-10)
                    if state[0] < Range['linear_z'][0] or state[0] > Range['linear_z'][1]:
                            crash_number += 1
            crash_count[i] = crash_number / demodata_.shape[0]
            average_error[i] = total_e / demodata_.shape[0]
        np.savetxt(agent_file_name, np.vstack((crash_count, average_error)), delimiter=',')
    else:
        crash_count = np.genfromtxt(agent_file_name, delimiter=',')[0]
        average_error = np.genfromtxt(agent_file_name, delimiter=',')[1]
    default_score = -K1 * crash_count - K2 * average_error
    
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(111)
    ax.plot(score[0], label='regular policy')
    ax.plot(score[1,0:], label='mixed non-deterministic policy')
    ax.plot(score[2,0:], label='mixed deterministic policy')
    ax.plot(np.ones_like(score[0])*demo_score, '--', label='demo')
    #ax.plot(default_score, '--', label='designed reward')

    ax.set_xlabel('iterations')
    ax.set_ylabel('scores')
    ax.set_xticks(np.arange(len(score[0])))
    ax.grid()
    ax.legend()
    '''
    if PLOT:
        plt.show()

if __name__ == '__main__':
    DIR = 'AL_results/projection_bad/projection_bad0/'
    control_eval(DIR, PLOT=True)
import numpy as np
import matplotlib.pyplot as plt

DIR_ = ['AL_results/projection_bad/projection_bad', 'AL_results/projection_good/projection_good',
    'AL_results/p_projection_bad/p_projection_bad', 'AL_results/p_projection_good/p_projection_good']
LABEL = ['AL-PFP with bad demo', 'AL-PFP with bad demo','SAL with bad demo', 'SAL with good demo']
fig2 = plt.figure(figsize=[10,5])
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('iterations')
ax2.set_ylabel('average tracking error')
for index, DIR in enumerate(DIR_):
    REPEATS = 10
    ITER = 20


    score = np.zeros(shape=[REPEATS, 3, ITER])
    for i in range(REPEATS):
        score[i, :, :] = np.genfromtxt(DIR + str(i) + '/score_file.csv', delimiter=',')[3:]

    score_aver = np.average(score, axis=0)
    score_error = np.std(score, axis=0)


    _, caps, bars = ax2.errorbar(np.arange(0, ITER), score_aver[2], capsize=3, yerr=score_error[0], label=LABEL[index])

    #_, caps, bars = ax2.errorbar(np.arange(0, ITER), score_aver[0], color='g', capsize=3, yerr=score_error[0], label='RL in SAL')
    # [bar.set_alpha(0.3) for bar in bars]
    # [cap.set_alpha(0.3) for cap in caps]

    # average tracking error for the best PID: -114.152
    # average tracking error for the bad PID: -283.5675
    # average tracking error for the RL: -193.727272
ax2.plot(np.ones_like(score_aver[0])*(114.152), '--', color='r', label='good demo', alpha=0.8)
ax2.plot(np.ones_like(score_aver[0])*(283.5675), '--', color='c', label='bad demo', alpha=0.8)
RL_score = np.genfromtxt('RL_results/RL_score.csv', delimiter=',')
ax2.plot(RL_score[1], '--', color='m', label='RL with handcrafted rewards', alpha=0.8)
ax2.set_xticks(np.arange(0, 20))
ax2.grid(linestyle=':')
fig2.legend(bbox_to_anchor=(0.9, 0.89))

plt.show()

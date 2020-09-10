import numpy as np
import matplotlib.pyplot as plt

# DIR = 'AL_results/projection_good_07noise/projection_good_07noise'
DIR = 'AL_results/projection_good/projection_good'
# DIR = 'AL_results/projection_good_09noise/projection_good_09noise'
REPEATS = 10
ITER = 20
mixed_fea = np.zeros(shape=[REPEATS, ITER, 462], dtype=float)
mixedfuzzy_fea = np.zeros(shape=[REPEATS, ITER, 462], dtype=float)
fea__ = np.zeros(shape=[REPEATS, ITER, 462], dtype=float)
fea = np.zeros(shape=[REPEATS, ITER, 462], dtype=float)
fea_e = np.zeros(shape=[REPEATS, 462], dtype=float)
dis_mixed = np.zeros(shape=[REPEATS, ITER])
dis_AL = np.zeros(shape=[REPEATS, ITER])
dis = np.zeros(shape=[REPEATS, ITER])


for i in range(REPEATS):
    mixed_fea[i, :, :] = np.genfromtxt(DIR + str(i) + '/' + 'mixed_feature_expectations.csv', delimiter=',')
    mixedfuzzy_fea[i, :, :] = np.genfromtxt(DIR + str(i) + '/' + 'mixedfuzzy_feature_expectations.csv', delimiter=',')
    fea__[i, :, :] = np.genfromtxt(DIR + str(i) + '/' + 'feature_expectation_AL.csv', delimiter=',')
    fea_e[i, :] = np.genfromtxt(DIR + str(i) + '/' + 'feature_expectations.csv', delimiter=',')[0, :-1]
    fea[i, :, :] = np.genfromtxt(DIR + str(i) + '/' + 'feature_expectations.csv', delimiter=',')[1:, :-1]

    dis_mixed[i, :] = [np.linalg.norm(f - fea_e[i, :]) for f in mixed_fea[i, :, :]]
    dis_AL[i, :] = [np.linalg.norm(f - fea_e[i, :]) for f in fea__[i, :, :]]
    dis[i, :] = [np.linalg.norm(f - fea_e[i, :]) for f in fea[i, :, :]]
dis_mixed_aver = np.average(dis_mixed, axis=0)
dis_AL_aver = np.average(dis_AL, axis=0)
dis_aver = np.average(dis, axis=0)
error_mixed = np.std(dis_mixed, axis=0)
error_AL = np.std(dis_AL, axis=0)
error = np.std(dis, axis=0)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('distance to $\mu_D$')
ax1.set_xlabel('iterations')
ax1.errorbar(np.arange(0, ITER), dis_mixed_aver, capsize=3, yerr=error_mixed, label='AL-PFP')
ax1.errorbar(np.arange(0, ITER), dis_AL_aver, yerr=error_AL, capsize=3, label='AL-Proj')
ax1.errorbar(np.arange(0, ITER), dis_aver, color='g', yerr=error, capsize=3, label='AL-RL')
ax1.set_xticks(np.arange(0, 20))
fig1.legend(bbox_to_anchor=(0.9, 0.89))
ax1.grid(linestyle=':')

score = np.zeros(shape=[REPEATS, 3, ITER])
for i in range(REPEATS):
    score[i, :, :] = np.genfromtxt(DIR + str(i) + '/score_file.csv', delimiter=',')[3:]

score_aver = np.average(score, axis=0)
score_error = np.std(score, axis=0)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('iterations')
ax2.set_ylabel('average tracking error')
_, caps, bars = ax2.errorbar(np.arange(0, ITER), score_aver[2], capsize=3, yerr=score_error[0], label='AL-PFP')

_, caps, bars = ax2.errorbar(np.arange(0, ITER), score_aver[0], color='g', capsize=3, yerr=score_error[0], label='AL-RL')
# [bar.set_alpha(0.3) for bar in bars]
# [cap.set_alpha(0.3) for cap in caps]

# average tracking error for the best PID: -114.152
# average tracking error for the bad PID: -283.5675
# average tracking error for the RL: -193.727272
ax2.plot(np.ones_like(score_aver[0])*(114.152), '--', color='r', label='good demo', alpha=0.8)
ax2.plot(np.ones_like(score_aver[0])*(283.5675), '--', color='c', label='bad demo', alpha=0.8)
RL_score = np.genfromtxt('RL_results/RL_score.csv', delimiter=',')
ax2.plot(RL_score[1], '--', color='m', label='RL with handcrafted rewards', alpha=0.8)
#ax2.errorbar(np.arange(0, ITER), RL_score[1], capsize=3, yerr=RL_score[3], label='RL with handcrafted rewards',alpha=0.8)

ax2.set_xticks(np.arange(0, 20))
ax2.grid(linestyle=':')
fig2.legend(bbox_to_anchor=(0.9, 0.89))

plt.show()

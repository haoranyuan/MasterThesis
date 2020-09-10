import numpy as np
from matplotlib import pyplot as plt


DIR = 'AL_results/projection_1(693)'

feat_e = np.genfromtxt(DIR+'/feature_expectations.csv', delimiter=',')[0, :-1]
feat = np.genfromtxt(DIR+'/feature_expectations.csv', delimiter=',')[1:, :-1]
feat_AL = np.genfromtxt(DIR+'/feature_expectation_AL.csv', delimiter=',')
feat_mixed = np.genfromtxt(DIR+'/mixed_feature_expectations.csv', delimiter=',')
feat_mixedfuzzy = np.genfromtxt(DIR+'/mixedfuzzy_feature_expectations.csv', delimiter=',')

feat_ = feat[:]
feat_AL_ = feat_AL[1:]
feat_AL_ = feat_
dist_matrix = np.empty(shape=(len(feat_AL_), len(feat_)))

for i, feAL in enumerate(feat_AL_):
    dist_vector = [np.linalg.norm(feAL - fe_) for fe_ in feat_]
    dist_matrix[i, :] = dist_vector

ticks = ['$\mu$'+str(t) for t in np.arange(20)]
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(dist_matrix)
for i in range(dist_matrix.shape[0]):
    for j in range(dist_matrix.shape[1]):
        text = ax.text(j, i, '{:1.2f}'.format(dist_matrix[i, j]), ha='center', va='center', color='w')
ax.set_xticks(np.arange(len(ticks)))
ax.set_yticks(np.arange(len(ticks)))
ax.set_xticklabels(ticks)
ax.set_yticklabels(ticks)
ax.set_xlim([-0.5, 19.5])
ax.set_ylim([-0.5, 19.5])


fig1, ax1 = plt.subplots()
k = 10
ax1.plot(np.repeat(feat_e[k], len(feat[1:, k])))
ax1.plot(feat[1:, k], label='feat')
ax1.plot(feat_AL[1:, k], label='feat_AL')
ax1.plot(feat_mixedfuzzy[1:, k], label='feat_mixedfuzzy')
ax1.plot(feat_mixed[1:, k], label='feat_mixed')
ax1.legend()
# 20 iterations deviation
devi_feat = np.sum(np.absolute(feat-feat_e), axis=1)
max_devi_feat = max(devi_feat)
devi_feat = devi_feat / max_devi_feat
devi_feat_AL = np.sum(np.absolute(feat_AL-feat_e), axis=1)/max_devi_feat
devi_feat_mixedfuzzy = np.sum(np.absolute(feat_mixedfuzzy-feat_e), axis=1)/max_devi_feat
devi_feat_mixed = np.sum(np.absolute(feat_mixed-feat_e), axis=1)/max_devi_feat
fig2, ax2 = plt.subplots()
ax2.plot(devi_feat, label='feat')
ax2.plot(devi_feat_AL, label='feat_AL')
ax2.plot(devi_feat_mixedfuzzy, label='feat_mixedfuzzy')
ax2.plot(devi_feat_mixed, label='feat_mixed')
ax2.legend()

fig.tight_layout()
plt.show()





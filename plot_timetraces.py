import numpy as np
import matplotlib.pyplot as plt

DIR1 = 'policies/bad/agentdata_mp.csv'
DIR2 = 'policies/good/agentdata_mp.csv'
DIR3 = 'policies/p_bad/agentdata_mp.csv'
DIR4 = 'policies/p_good/agentdata_mp.csv'

DEMO1 = 'AL_results/Demonstrations/demodata_bad_headers.csv'
DEMO2 = 'AL_results/Demonstrations/demodata_best_headers.csv'

index = 10
traj1 = np.reshape(np.genfromtxt(DIR1, delimiter=','), newshape=[-1, 201, 3])[index, :, :2]
traj2 = np.reshape(np.genfromtxt(DIR2, delimiter=','), newshape=[-1, 201, 3])[index, :, :2]
traj3 = np.reshape(np.genfromtxt(DIR3, delimiter=','), newshape=[-1, 201, 3])[index, :, :2]
traj4 = np.reshape(np.genfromtxt(DIR4, delimiter=','), newshape=[-1, 201, 3])[index, :, :2]

demo1 = np.reshape(np.genfromtxt(DEMO1, delimiter=','), newshape=[-1, 201, 4])[index, :, :2]
demo2 = np.reshape(np.genfromtxt(DEMO2, delimiter=','), newshape=[-1, 201, 4])[index, :, :2]

fig1 = plt.figure(figsize=[10,7])
ax1 = fig1.add_subplot(211)
ax1.plot(traj1[:, 0], label='AL with bad demo')
ax1.plot(demo1[:, 0], linestyle='-.', label='bad demo')
ax1.plot(traj2[:, 0], label='AL with good demo')
ax1.plot(demo2[:, 0], linestyle='-.', label='good demo')
ax1.plot(0.1428*np.ones_like(traj1[:, 0]), 'k--')
ax1.plot(-0.1428*np.ones_like(traj1[:, 0]), 'k--')
ax1.set_xlim([0, 200])
ax1.legend()
ax1.grid(linestyle=':')
ax1.set_ylabel('Position (m)')
ax1.set_xlabel('Time steps')


ax1 = fig1.add_subplot(212)
ax1.plot(traj1[:, 1], label='AL with bad demo')
ax1.plot(demo1[:, 1], linestyle='-.', label='bad demo')
ax1.plot(traj2[:, 1], label='AL with good demo')
ax1.plot(demo2[:, 1], linestyle='-.', label='good demo')
ax1.plot(0.045*np.ones_like(traj1[:, 0]), 'k--')
ax1.plot(-0.045*np.ones_like(traj1[:, 0]), 'k--')
ax1.set_xlim([0, 200])
ax1.set_ylabel('Velocity (m$\cdot$s$^{-1}$)')
ax1.set_xlabel('Time steps')
ax1.grid(linestyle=':')
ax1.legend()
plt.show()
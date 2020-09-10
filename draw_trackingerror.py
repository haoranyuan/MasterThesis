import numpy as np
from demo_discrete import Discretize
from matplotlib import pyplot as plt


demo_data = np.genfromtxt('AL_results/projection_1/demodata.csv', delimiter=',')
sa = demo_data[:, :-1]
dis = Discretize(data=sa)
#dis_demo = dis.discretize_data()
#dis_sa = dis.discretize_data()
ep_len = 200

error = np.absolute(sa[:, 0]).reshape([-1, 200])
#dis_error = np.absolute(dis_sa[:, 0]-10).reshape([-1, 200])

mean_error = np.mean(error, axis=0)
#mean_dis_error = np.mean(dis_error, axis=0)

agent_data = np.genfromtxt('AL_results/projection_1/iter12/agentdata_AL.csv', delimiter=',')
error_agent = np.absolute(agent_data[:, 0]).reshape([-1, 200])
mean_error_agent = np.mean(error_agent, axis=0)

agentQ_data = np.genfromtxt('agentdata_default.csv', delimiter=',')[:200]
#error_agentQ = np.absolute(agentQ_data).reshape([-1, 200])
#mean_error_agentQ = np.mean(error_agentQ, axis=0)

#dis.data = agent_data
#dis_agent = dis.discretize_data()
#dis.data = agentQ_data
#dis_agnetQ = dis.discretize_data()


fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = fig.add_subplot(122)
#ax1.plot(mean_error)
ax1.plot(sa[0:3000, 0])
ax1.plot(agent_data[0:3000, 0])
#ax1.plot(dis_agnetQ[:, 0])
#ax2.plot(mean_dis_error)
ax1.grid()
#ax2.grid()

fig2 = plt.figure(figsize=[5, 12])
ax2 = fig2.add_subplot(111)
i = 2000
ax2.plot(sa[i+200:i+400, 1], sa[i+200:i+400, 0])
ax2.plot(agent_data[i:i+200, 1], agent_data[i:i+200, 0], alpha=0.8, linestyle='--')
ax2.set_xlim([-0.52, 0.52])
ax2.set_ylim([-3, 3])

ax2.grid()

plt.show()



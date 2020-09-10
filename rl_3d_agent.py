import numpy as np
from os.path import isfile


class errorlog():
    def __init__(self, filename):
        self.filename = filename
        if isfile(self.filename):
            print('loading error log')
            e_rate = np.genfromtxt(self.filename, delimiter=',')
            self.e = e_rate[0,:]
            self.success_rate = e_rate[1,:]
        else:
            self.e = []
            self.success_rate = []

    def collect(self, e_entries, success_rate):
        self.e = np.append(self.e, e_entries)
        self.e = list(self.e)
        self.success_rate = np.append(self.success_rate, success_rate)
        self.success_rate = list(self.success_rate)

    def save_e(self):
        e_r = np.asarray([self.e,self.success_rate])
        np.savetxt(self.filename, e_r, delimiter=',')


class RL_3D():

    def __init__(self, change_sav, get_sav, params,
                 lr=0.02, epsilon=0.8, gamma=0.9, policy_dir=None):

        self.thread_object = None
        self.state_log = []
        self.log = np.zeros([1,8])
        self.E = epsilon
        self.change_sav = change_sav
        self.get_sav = get_sav
        self.optimal = 0
        self.target_state = params['q1']['target_state']
        self.sav_error = 0
        self.state_action_num = params['q1']['sa_num']
        self.terminal_states = params['q1']['terminal']
        self.ep_s = []
        self.ep_r = []
        self.ep_a = []
        self.ep_s_ = []
        self.statecontlog = []
        self.statecont_log = []
        self.actioncont_log = []
        self.lr = lr
        self.gamma = gamma
        self.tem = 0.99
        self.POLICY = None
        if policy_dir is not None:
            self.POLICY = np.genfromtxt(policy_dir, delimiter=',')


    def learn(self):
        s = self.ep_s[-1]
        s_ = self.ep_s_[-1]
        a = self.ep_a[-1]
        r = self.ep_r[-1]

        sav = self.get_sav()
        q_predict = sav[tuple(np.concatenate((s, a)))]
        # Use this when the task has terminal states
        #if list(s_) in self.terminal_states:
            #update_target = r
        #else:
            #update_target = r + self.gamma * max(sav[tuple(s_)])
        #Use this when the task is continuous, or has fix episode length
        update_target = r + self.gamma * max(sav[tuple(s_)])
        update = q_predict + self.lr * (update_target - q_predict)
        self.change_sav(index=np.concatenate((s, a)),
                        value=update)

    def store_trans(self, s, a, r, s_, scont=None, scont_=None, acont=None):
        # s, a, s_ are numpy arrays
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)
        self.ep_s_.append(s_)
        self.statecontlog.append(scont)
        self.statecont_log.append(scont_)
        self.actioncont_log.append(acont)

    def policy(self, state):
        state = tuple(state)
        if self.POLICY is not None:
            a = self.policy_direct(state)
        else:
            sav = self.get_sav()
            sav_action = sav[state]
            # An epsilon-greedy policy
            where_max = np.where(sav_action == np.amax(sav_action))
            prob = (1-self.E)/sav_action.size*np.ones_like(sav_action)
            prob[where_max] += self.E/len(where_max[0])
            # alternative policy
            #soft_max = [np.e**(i/self.tem) for i in sav_action]
            #prob = [i/np.sum(soft_max) for i in soft_max]
            a = np.random.choice(len(sav_action), size=1, p=prob)

        return a

    def policy_direct(self, state):
        # loading policy
        prob = [1-self.POLICY[state], self.POLICY[state]]
        a = np.array(np.random.choice(a=[0, 1], size=1, p=prob))
        return a

    def get_log(self):
        self.log = np.reshape(np.asarray(self.state_log), (-1, 3))

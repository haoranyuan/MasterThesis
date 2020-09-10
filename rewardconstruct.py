import numpy as np


class feature:
    def __init__(self):
        self.state_action_num = [21, 11, 2]
        self.target = [10, 5]

        span = []
        for idx in range(len(self.state_action_num)):
            i = 1
            if idx + 1 < len(self.state_action_num):
                for j in self.state_action_num[idx + 1:]:
                    i = i * j
            span.append(i)
        self.span = span

    def get_feature(self, dis_state_action):
        i = 1
        for j in self.state_action_num:
            i = i * j
        feature = np.zeros(i)
        feature_code = int(np.dot(list(dis_state_action), self.span))
        feature[feature_code] = 1
        return feature


class FeatureExpectation:
    def __init__(self):
        self.state_action_num = [21, 11, 2]
        self.target = [10, 5]
        self.feature = feature()

    def featureexpectations(self, trajectories, ep_len=200, DISCOUNT = 0.9, header_exist=False):
        self.ep_len = ep_len
        traj_count = 0
        discount = 1
        i = 1
        for j in self.state_action_num:
            i = i * j
        EXPECTATION = np.zeros(i)
        if header_exist:
            trajs = np.reshape(trajectories, newshape=[-1, ep_len+1, trajectories.shape[1]])
        else:
            trajs = np.reshape(trajectories, newshape=[-1, ep_len, trajectories.shape[1]])
        for traj in trajs:
            if header_exist:
                traj = traj[1:]
            traj_count += 1
            discount = 1
            for realization in traj:
                feature = self.feature.get_feature(realization)
                EXPECTATION = EXPECTATION + discount * np.asarray(feature)
                discount = discount * DISCOUNT
        # feature occupancy
        EXPECTATION = EXPECTATION / traj_count / (1 - DISCOUNT**self.ep_len) * (1-DISCOUNT) * 100
        self.expectation = EXPECTATION
        print('Feature: traj count: ', traj_count)
        return EXPECTATION, traj_count


class Rewardconstruct:
    def __init__(self):
        self.state_action_num = [21, 11, 2]
        self.feature = feature()
        self.rewardscheme = np.zeros(self.state_action_num,dtype=float)

    def reward_scheme(self,omega,scale):
        self.omega = omega
        i = 1
        for j in self.state_action_num:
            i = i * j
        for index in range(i):
            dis_state_action = tuple(np.unravel_index(index, self.state_action_num))
            fea = self.feature.get_feature(dis_state_action)
            r = np.dot(fea, self.omega)
            self.rewardscheme[dis_state_action] = r
        self.rewardscheme = self.rewardscheme * scale
        return self.rewardscheme


class CustomizeReward:
    def __init__(self):
        self.state_action_num = [21, 11, 2]
        self.target = tuple([10, 5])
        self.rewardscheme = np.zeros(self.state_action_num)

    def get_reward_scheme(self, plan_id):
        for y in range(self.state_action_num[0]):
            for y_d in range(self.state_action_num[1]):
                for p1 in range(self.state_action_num[2]):
                        idx = (y, y_d, p1)
                        self.rewardscheme[idx] = self.rewardplan(idx,plan_id)
        self.rewardscheme = self.rewardscheme.reshape([462, ])
        if plan_id == 'B':
            factor = 3
        else:
            factor = 1
        self.rewardscheme = factor * self.rewardscheme / abs(min(self.rewardscheme))
        np.savetxt('reward_'+plan_id+'.csv',self.rewardscheme,delimiter=',')

    def rewardplan(self,idx,plan_id):
        dist = np.array(idx[0:len(self.target)]) - np.array(self.target)
        if plan_id == 'default':
            r = -(1 * (abs(dist[0])**2) + 1 * (abs(dist[1])**1))
        if plan_id == 'A':
            if idx[0:len(self.target)] == self.target:
                r = 0
            else:
                r = -1
        if plan_id == 'Yuri':
            e = 1e-3
            r = - 100 * (dist[0]*dist[1]) / (abs(dist[0]*dist[1])+e) / (0.5 - idx[-1]) + 1/ (abs(dist[0])+e) - 1/e
        if plan_id == 'B':
            r = - 3 * (abs(dist[0]))
        return r



if __name__ == '__main__':
    R = CustomizeReward()
    plan = 'B'
    R.get_reward_scheme(plan_id=plan)
    #R.get_reward_scheme(plan_id='A')
    r = np.genfromtxt('reward_'+plan+'.csv', delimiter=',')
    #r = np.genfromtxt('reward_A.csv', delimiter=',')
    print(np.unravel_index(np.argmax(r), [21, 11, 2]))
import numpy as np

class Discretize():
    def __init__(self, data=None):
        self.data = data
        Range = {
            'linear_z': [-3, 3],
            'linear_zrate': [-0.5, 0.5],
            'Motor': [0, 1]
        }

        action_num = np.int32(2)
        state_num = np.array([21, 11], dtype=np.int32)
        state_action_num = np.hstack((state_num, action_num))
        target_state = np.array([10, 5], dtype=np.int32)
        self.range = Range
        self.state_action_num = state_action_num
        self.target_state = target_state

    def discretize_data(self):
        ds_data = np.zeros_like(self.data, dtype=np.int32)
        for j, state in enumerate(self.data):
            t = np.array([0, 0])
            state[0:-1] = state[0:-1] - t

            for i, key in enumerate(self.range):
                margin = (self.range[key][1] - self.range[key][0]) / self.state_action_num[i]
                ds_data[j][i] = int((state[i] - self.range[key][0]) // margin)
                ds_data[j][i] = np.clip(ds_data[j][i], 0, self.state_action_num[i] - 1)
            '''
            for i, key in enumerate(self.range):
                s_abs = abs(state[i])
                s0 = 1/(2**((self.state_action_num[i] - 2 -1) / 2 +2) - 3) * (self.range[key][1] - self.range[key][0])
                j = 1
                while s0*(2**j - 1) - 0.5*s0 < s_abs:
                    j += 1
                j += -1
                j = np.clip(j, 0, (self.state_action_num[i] -1) / 2)
                j = j * state[i]/s_abs
                j = j + (self.state_action_num[i] -1) / 2
                ds[i] = j
            '''
        return ds_data

    def _get_action(self, motor):
        ac = np.zeros_like(motor, dtype=float)
        for i, m in enumerate(motor):
            ac[i] = round((m - self.range['Motor'][0])/(self.range['Motor'][1] - self.range['Motor'][0]))
        return ac
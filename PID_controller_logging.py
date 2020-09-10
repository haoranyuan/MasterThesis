import numpy as np
import math
import time
import threading
from os.path import isfile

class errorlog():
    def __init__(self, filename='e_demo.csv'):
        self.filename = filename
        if isfile(self.filename):
            self.success_rate = np.genfromtxt(self.filename, delimiter=',')
        else:
            self.success_rate = []

    def collect(self, success_rate):
        self.success_rate = np.append(self.success_rate, success_rate)
        self.success_rate = list(self.success_rate)

    def save_e(self):
        e_r = np.asarray([self.success_rate])
        np.savetxt(self.filename, e_r, delimiter=',')

    # def draw_e(self):



class PID_3D():

    def __init__(self, params, quad_params, ctrl_noise):

        self.target = [0, 0, 0]
        self.yaw_target = 0.0
        self.MOTOR_LIMITS = params['Motor_limits']
        self.TILT_LIMITS = [(params['Tilt_limits'][0] / 180.0) * 3.1416, (params['Tilt_limits'][1] / 180.0) * 3.1416]
        self.YAW_CONTROL_LIMITS = params['Yaw_Control_Limits']
        self.Z_LIMITS = [self.MOTOR_LIMITS[0] + params['Z_XY_offset'], self.MOTOR_LIMITS[1] - params['Z_XY_offset']]
        self.LINEAR_P = params['Linear_PID']['P']
        self.LINEAR_I = params['Linear_PID']['I']
        self.LINEAR_D = params['Linear_PID']['D']
        self.LINEAR_TO_ANGULAR_SCALER = params['Linear_To_Angular_Scaler']
        self.YAW_RATE_SCALER = params['Yaw_Rate_Scaler']
        self.ANGULAR_P = params['Angular_PID']['P']
        self.ANGULAR_I = params['Angular_PID']['I']
        self.ANGULAR_D = params['Angular_PID']['D']
        self.xi_term = 0
        self.yi_term = 0
        self.zi_term = 0
        self.thetai_term = 0
        self.phii_term = 0
        self.gammai_term = 0
        self.thread_object = None
        self.state_log = []
        self.ep_s = []
        self.ep_r = []
        self.ep_a = []
        self.ep_s_ = []
        self.statecontlog = []
        self.statecont_log = []
        self.actioncont_log = []
        self.range = quad_params['q1']['range']
        self.state_action_num = quad_params['q1']['sa_num']
        self.target_state = quad_params['q1']['target_state']
        self.ctrl_noise = ctrl_noise
        # state_action value function (sav) is constructed as a discrete state-action space with (state_action_num[0]+2)
        # *(state_action_num[1] + 2)* ... states. The action space is limited by motor limits

    def deg2arc(self, deg):
        return np.pi*deg/180

    def wrap_angle(self, val):
        return ((val + np.pi) % (2 * np.pi) - np.pi)

    def update_target(self, target):
        self.target = target

    # RL agent control for tracking control
    # From A to B

    def PID_update(self, state):
        state = self.dis2cont(state)
        z, z_dot = state
        z_error = 0-z
        self.xi_term += 0
        self.yi_term += 0
        self.zi_term += self.LINEAR_I[2]*z_error
        dest_z_dot = self.LINEAR_P[2]*(z_error) + self.LINEAR_D[2]*(-z_dot) + self.zi_term
        throttle = np.clip(dest_z_dot,self.Z_LIMITS[0],self.Z_LIMITS[1])
        self.thetai_term += 0
        self.phii_term += 0
        self.gammai_term += 0
        y_val = 0
        z_val = self.ANGULAR_P[2]*0 + self.gammai_term
        z_val = 0
        # m1 = throttle + x_val + z_val
        m2 = throttle
        # m3 = throttle - x_val + z_val
        m4 = throttle
        m1 = (m2 + m4)/2
        M = np.clip([m2,m4],self.MOTOR_LIMITS[0],self.MOTOR_LIMITS[1])
        M_dis = np.array([self.motor2action(M[0], motor_num=1)])
        if np.random.rand() > self.ctrl_noise:
            M_dis = np.abs(M_dis-1)
        #print(state, M_dis)
        return M_dis

    def policy(self, state):
        PID_dis_ac = self.PID_update(state)
        return PID_dis_ac

    def dis2cont(self, dis_s):
        t = np.array([0, 0])
        state_num = np.array(self.state_action_num[0:-1])
        state = np.zeros_like(state_num, dtype = float)
        for i, key in enumerate(self.range):
            margin = (self.range[key][1] - self.range[key][0])/(state_num[i])
            state[i] = self.range[key][0] + (dis_s[i] + 0.5) * margin
        state = state + t
        return state

    def motor2action(self,motor, motor_num):
        action_margin = (self.MOTOR_LIMITS[1] - self.MOTOR_LIMITS[0]) / self.state_action_num[-2 + motor_num]
        ac = (motor - self.MOTOR_LIMITS[0]) // action_margin
        ac = np.clip(ac, 0, self.state_action_num[-2+motor_num]-1)
        ac = int(ac)
        return ac

    def store_trans(self, s, a, r, s_, scont, scont_, acont):
        # s, a, s_ are numpy arrays
        #self.ep_s.append(s)
        #self.ep_a.append(a)
        self.ep_r.append(r)
        self.ep_s_.append(s_)
        self.statecontlog.append(scont)
        self.statecont_log.append(scont_)
        self.actioncont_log.append(acont)
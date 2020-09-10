import numpy as np
import math
import time
import threading


class RL_3D():

    def __init__(self, change_sav, get_sav,  get_state, get_state_action, get_time, actuate_motors, params, quad_identifier, restart_quads):

        self.timer = 0
        self.get_state = get_state
        self.quad_identifier = quad_identifier
        self.actuate_motors = actuate_motors
        self.get_state_action = get_state_action
        self.get_time = get_time
        self.restart = restart_quads
        self.thread_object = None
        self.target = [0, 0, 0]
        self.yaw_target = 0.0
        self.run = True
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

        self.E = 0.9
        self.range = { 'linear_x': [-3,3],
                       'linear_z': [-3, 3],
                       'linear_xrate': [-0.5,0.5],
                       'linear_zrate': [-0.5, .5],
                       'pitch_angle': self.TILT_LIMITS,
                       'pitch_angle_rate': [self.deg2arc(-0.3), self.deg2arc(.3)]
        }
        self.filename = 'sav_3d.csv'
        action_num = 5
        state_num = np.array([3, 3, 1, 1, 3, 1]) + 2
        self.state_action_num = tuple(np.append(state_num, [action_num, action_num]))
        self.change_sav = change_sav
        self.get_sav = get_sav
        self.last_sa = self.get_state_action(self.quad_identifier)
        # state_action value function (sav) is constructed as a discrete state-action space with (state_action_num[0]+2)
        # *(state_action_num[1] + 2)* ... states. The action space is limited by motor limits

    def deg2arc(self, deg):
        return np.pi*deg/180

    def wrap_angle(self, val):
        return ((val + np.pi) % (2 * np.pi) - np.pi)

    def update_target(self, target):
        self.target = target

    def update_yaw_target(self, target):
        self.yaw_target = self.wrap_angle(target)

    def thread_run(self,update_rate,sav_rate,time_scaling):
        update_rate = update_rate*time_scaling
        sav_rate = sav_rate * time_scaling
        last_update = self.get_time()
        last_save = self.get_time()
        while(self.run==True):
            time.sleep(0)
            self.time = self.get_time()
            if (self.time - last_update).total_seconds() > update_rate:
                self.update_action()
            #    last_update = self.time
            # if (self.time - last_update).total_seconds() > sav_rate:
                self.sav_update()
                self.last_sa = self.get_state_action(self.quad_identifier)
                last_update = self.time
            if (self.time - last_save).total_seconds() > 100*update_rate:
                if self.isfinished():
                    self.save_file(self.filename)
                    self.restart(self.quad_identifier)
                    self.last_sa = self.get_state_action(self.quad_identifier)
                last_save = self.time

    def start_thread(self, update_rate=0.005, time_scaling=1):
        sav_rate = update_rate
        self.thread_object = threading.Thread(target=self.thread_run, args=(update_rate, sav_rate, time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

    # RL agent control for tracking control
    # From A to B
    def update_action(self):
        [x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot, m1, m2, m3, m4] = self.get_state_action(
            self.quad_identifier)
        s = [x,z,x_dot,z_dot, phi, phi_dot]
        ac = [m2, m4]
        sa = s + ac
        ds,_ = self.cont2dis(state_action=sa)
        ac_new, PID_flag, PID_ac_new = self.policy(self.E, ds)

        if not PID_flag:
            m2 = self.action2motor(ac_new[0], 1)
            m4 = self.action2motor(ac_new[1], 2)
            # print('RL action=',[m2, m4])
        else:
            m2,m4 = PID_ac_new
            #print('PID action=', [m2, m4])
        m2, m4 = np.clip([m2, m4], self.MOTOR_LIMITS[0], self.MOTOR_LIMITS[1])
        m1 = (m2+m4)/2
        m3 = m1
        M =  [m1, m2, m3, m4]
        self.actuate_motors(self.quad_identifier, M)

    def PID_update(self):
        # 2-D tracking requires only the x-z plane variables
        # des_y, dest_ydot, dest_theta, dest_yaw and all the other states that exist in other planes should be zeros
        [dest_x,dest_y,dest_z] = self.target
        [x,y,z,x_dot,y_dot,z_dot,theta,phi,gamma,theta_dot,phi_dot,gamma_dot] = self.get_state(self.quad_identifier)
        x_error = dest_x-x
        y_error = dest_y-y
        z_error = dest_z-z
        self.xi_term += self.LINEAR_I[0]*x_error
        self.yi_term += self.LINEAR_I[1]*y_error
        self.zi_term += self.LINEAR_I[2]*z_error
        dest_x_dot = self.LINEAR_P[0]*(x_error) + self.LINEAR_D[0]*(-x_dot) + self.xi_term
        dest_y_dot = self.LINEAR_P[1]*(y_error) + self.LINEAR_D[1]*(-y_dot) + self.yi_term
        dest_z_dot = self.LINEAR_P[2]*(z_error) + self.LINEAR_D[2]*(-z_dot) + self.zi_term
        throttle = np.clip(dest_z_dot,self.Z_LIMITS[0],self.Z_LIMITS[1])
        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0]*(dest_x_dot*math.sin(gamma)-dest_y_dot*math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1]*(dest_x_dot*math.cos(gamma)+dest_y_dot*math.sin(gamma))
        dest_gamma = self.yaw_target
        dest_theta,dest_phi = np.clip(dest_theta,self.TILT_LIMITS[0],self.TILT_LIMITS[1]),np.clip(dest_phi,self.TILT_LIMITS[0],self.TILT_LIMITS[1])
        theta_error = dest_theta-theta
        phi_error = dest_phi-phi
        gamma_dot_error = (self.YAW_RATE_SCALER*self.wrap_angle(dest_gamma-gamma)) - gamma_dot
        self.thetai_term += self.ANGULAR_I[0]*theta_error
        self.phii_term += self.ANGULAR_I[1]*phi_error
        self.gammai_term += self.ANGULAR_I[2]*gamma_dot_error
        x_val = self.ANGULAR_P[0]*(theta_error) + self.ANGULAR_D[0]*(-theta_dot) + self.thetai_term
        y_val = self.ANGULAR_P[1]*(phi_error) + self.ANGULAR_D[1]*(-phi_dot) + self.phii_term
        z_val = self.ANGULAR_P[2]*(gamma_dot_error) + self.gammai_term
        z_val = np.clip(z_val,self.YAW_CONTROL_LIMITS[0],self.YAW_CONTROL_LIMITS[1])
        # m1 = throttle + x_val + z_val
        m2 = throttle + y_val - z_val
        # m3 = throttle - x_val + z_val
        m4 = throttle - y_val - z_val
        m1 = (m2 + m4)/2
        m3 = m1
        M = np.clip([m2,m4],self.MOTOR_LIMITS[0],self.MOTOR_LIMITS[1])
        M_dis = [self.motor2action(M[0],motor_num=1), self.motor2action(M[1],motor_num=2)]
        return M_dis, M



    def sav_update(self):
        # print("updateing sav...")
        sav = self.get_sav()
        [x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot, m1, m2, m3,
         m4] = self.get_state_action(
            self.quad_identifier)
        s_t = [x, z, x_dot, z_dot, phi, phi_dot]
        ac_t = [m2, m4]
        state_action_t = s_t + ac_t
        # print("first cont2dis in update sav")
        state_t, _ = self.cont2dis(state_action_t)
        a_t,_,_ = self.policy(1, state_t) # Q-learning update the best next-state-action value
        # print(state_t,a_t)
        value_t = sav[state_t+a_t]

        [x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot, m1, m2, m3,
         m4] = self.last_sa
        s = [x, z, x_dot, z_dot, phi, phi_dot]
        ac = [m2, m4]
        state_action = s + ac
        # print("second cont2dis in update sav")
        s,a = self.cont2dis(state_action)
        value = sav[s+a]
        # Now update the last state-action value function
        # self.sav[s+a] = value + 0.001*(self.reward(state_t+a_t) + 0.999*value_t)
        self.change_sav(index=s+a,
                        value=value + 0.001 * (self.reward(state_t + a_t) + 0.999 * value_t))



    def policy(self, E, state):
        sav = self.get_sav()
        # An epsilon-greedy policy
        epsilon = np.random.random()
        action_max = np.unravel_index(np.argmax(sav[state], axis=None), sav[state].shape)
        action = np.unravel_index(np.argmax(sav[state], axis=None), sav[state].shape)
        PID_action = np.zeros_like(action)
        PID_flag = 0
        action,PID_action = self.PID_update()
        PID_flag = 1
        return tuple(action), PID_flag,PID_action

    def motor2action(self,motor, motor_num):
        action_margin = (self.MOTOR_LIMITS[1] - self.MOTOR_LIMITS[0]) / self.state_action_num[-3 + motor_num]
        ac = (motor - self.MOTOR_LIMITS[0]) // action_margin
        ac = np.clip(ac, 0, self.state_action_num[-3+motor_num]-1)
        ac = int(ac)
        return ac

    def cont2dis(self, state_action):
        #print("in cont2dis")
        state = np.array(state_action[0:-2])
        t = np.array([self.target[0], self.target[2], 0, 0, 0, 0])
        state = state - t
        action = state_action[-2:]
        state_num = np.array(self.state_action_num[0:-2])
        action_num = np.array(self.state_action_num[-2:])
        ds = np.zeros_like(state,dtype=int)
        for i, key in enumerate(self.range):
            # r = self.range[key]
            margin = (self.range[key][1]-self.range[key][0])/(state_num[i]-2)
            if state[i] < self.range[key][0]:
                ds[i] = 0
                continue
            if state[i] >= self.range[key][1]:
                ds[i] = state_num[i] - 1
                continue
            while self.range[key][0] + (ds[i]) * margin <= state[i] and ds[i] < state_num[i]-1:
                ds[i] += 1

        dis_ac = np.zeros_like(action,dtype=int)
        for i, ac in enumerate(action):
            dis_ac[i] = self.motor2action(ac, i+1)
        dis_sa_tuple = [tuple(ds), tuple(dis_ac)]
        #print('origin sa:', state_action)
        #print('dis sa: ', dis_sa_tuple)
        return dis_sa_tuple

    def action2motor(self,action, motor_num):
        action_margin = (self.MOTOR_LIMITS[1] - self.MOTOR_LIMITS[0]) / self.state_action_num[-3 + motor_num]
        speed = action*action_margin + self.MOTOR_LIMITS[0]
        return speed

    def save_file(self,file_name):
        SAV = self.get_sav()
        dim = 1
        for d in self.state_action_num:
            dim = dim * d
        SAV = SAV.reshape([dim, 1])
        np.savetxt(file_name, SAV, delimiter=',')

    def isfinished(self):
        dis_target = np.array([(self.state_action_num[i] - 1) / 2 for i in range(len(self.state_action_num))])
        [x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot, m1, m2, m3,
         m4] = self.get_state_action(
            self.quad_identifier)
        state_action = np.array([x, z, x_dot, z_dot, phi, phi_dot, m2, m4])
        dis_state,_ = np.array(self.cont2dis(state_action))
        if dis_state == dis_target:
            flag = 1
        else:
            target = np.array([self.target[0], self.target[2], 0, 0, 0, 0])
            if np.linalg.norm(state_action[0:-2]-target) > 10:
                flag = 1
            else:
                flag = 0
        return flag

    def killall(self):
        self.stop_thread()

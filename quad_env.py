import numpy as np
import math
import scipy.integrate
from numpy.random import uniform
from os.path import isfile
import threading


class Propeller():
    def __init__(self, prop_dia, prop_pitch, thrust_unit='N'):
        self.dia = prop_dia
        self.pitch = prop_pitch
        self.thrust_unit = thrust_unit
        self.speed = 5300 #RPM
        self.thrust = 0

    def set_speed(self,speed):
        self.speed = speed
        # From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
        self.thrust = 4.392e-8 * self.speed * math.pow(self.dia,3.5)/(math.sqrt(self.pitch))
        self.thrust = self.thrust*(4.23e-4 * self.speed * self.pitch)
        if self.thrust_unit == 'Kg':
            self.thrust = self.thrust*0.101972

class Quadcopter():
    # State space representation: [x y z x_dot y_dot z_dot theta phi gamma theta_dot phi_dot gamma_dot]
    def __init__(self, quads, motorlimits, gravity=9.81, b=0.0245, rewardflag='default', file_dir='results'):
        self.quads = quads
        self.g = gravity
        self.b = b
        self.MOTOR_LIMITS = motorlimits
        self.ode =  scipy.integrate.ode(self.state_dot).set_integrator('vode',nsteps=500,method='bdf')
        self.init = {}
        key = 'q1'
        self.quads[key]['state'] = np.zeros(12)
        self.quads[key]['state'][0:3] = self.quads[key]['position']
        self.quads[key]['state'][6:9] = self.quads[key]['orientation']
        self.quads[key]['m1'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
        self.quads[key]['m2'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
        self.quads[key]['m3'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
        self.quads[key]['m4'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
        ixx=((2*self.quads[key]['weight']*self.quads[key]['r']**2)/5)+(2*self.quads[key]['weight']*self.quads[key]['L']**2)
        iyy=ixx
        izz=((2*self.quads[key]['weight']*self.quads[key]['r']**2)/5)+(4*self.quads[key]['weight']*self.quads[key]['L']**2)
        self.quads[key]['I'] = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
        self.quads[key]['invI'] = np.linalg.inv(self.quads[key]['I'])
        self.init[key] = {}
        self.init[key]['state'] = self.quads[key]['state']
        self.init[key]['m1_speed'] = self.quads[key]['m1'].speed
        self.init[key]['m2_speed'] = self.quads[key]['m1'].speed
        self.init[key]['m3_speed'] = self.quads[key]['m1'].speed
        self.init[key]['m4_speed'] = self.quads[key]['m1'].speed
        self.range = self.quads[key]['range']
        self.state_action_num = self.quads[key]['sa_num']
        self.target_state = self.quads[key]['target_state']
        np.random.seed(10)
        self.resetstate = np.vstack((np.repeat(np.arange(self.state_action_num[0]), self.state_action_num[1]).reshape((-1, self.state_action_num[1])).transpose().reshape((self.state_action_num[0]*self.state_action_num[1]))
,
                                     np.repeat(np.arange(self.state_action_num[1]), self.state_action_num[0]).reshape((-1, self.state_action_num[0])).transpose().reshape((self.state_action_num[0]*self.state_action_num[1]))
                                    )).transpose()
        '''
        self.resetstate = np.vstack((np.repeat(np.arange(self.state_action_num[0]), self.state_action_num[1]).reshape(
            (-1, self.state_action_num[1])).transpose().reshape((self.state_action_num[0] * self.state_action_num[1]))
                                     ,
                                     np.repeat(np.arange(self.state_action_num[1]), self.state_action_num[0])
                                     )).transpose()
        '''
        self.resetstate = np.vstack((np.repeat(np.arange(self.state_action_num[0]), self.state_action_num[1]),
                                     np.repeat(np.arange(self.state_action_num[1]), self.state_action_num[0]).reshape(
                                         (-1, self.state_action_num[0])).transpose().reshape(
                                         (self.state_action_num[0] * self.state_action_num[1]))
                                     )).transpose()
        # Alternate choice: delete some part of trajectory set
        half = 0
        if half == 1:
            # (1) keep the first half
            self.resetstate = self.resetstate[:int(len(self.resetstate)/2), :]
        elif half == 2:
            # (2) keep the second half
            self.resetstate = self.resetstate[int(len(self.resetstate)/2):, :]
        elif half == 3:
            # (3) keep the random half
            np.random.shuffle(self.resetstate)
            self.resetstate = self.resetstate[:int(len(self.resetstate)/2), :]

        #np.random.shuffle(self.resetstate)
        self.reset_pointer = 0

        if isfile(file_dir+'reward_'+rewardflag+'.csv'):
            print('Quad: loading reward function')
            self.rewardscheme = np.genfromtxt(file_dir+'reward_'+rewardflag+'.csv', delimiter=',')
            self.rewardscheme = np.reshape(self.rewardscheme, self.state_action_num)
        else:
            print('Quad: Did not find ', file_dir+'reward_'+rewardflag+'.csv', ' loading default reward instead' )
            self.rewardscheme = np.genfromtxt('reward_default.csv', delimiter=',')
            self.rewardscheme = np.reshape(self.rewardscheme, self.state_action_num)

    def rotation_matrix(self,angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def wrap_angle(self,val):
        return( ( val + np.pi) % (2 * np.pi ) - np.pi )

    def state_dot(self, time, state, key='q1'):
        state_dot = np.zeros(12)
        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0] = 0
        state_dot[1] = 0
        state_dot[2] = self.quads[key]['state'][5]
        # The acceleration
        x_dotdot = np.array([0,0,-self.quads[key]['weight']*self.g]) + np.dot(self.rotation_matrix(self.quads[key]['state'][6:9]),np.array([0,0,(self.quads[key]['m1'].thrust + self.quads[key]['m2'].thrust + self.quads[key]['m3'].thrust + self.quads[key]['m4'].thrust)]))/self.quads[key]['weight']
        state_dot[3] = 0
        state_dot[4] = 0
        state_dot[5] = x_dotdot[2]
        # The angular rates(t+1 theta_dots equal the t theta_dots)
        state_dot[6] = 0
        state_dot[7] = 0
        state_dot[8] = 0
        # The angular accelerations
        omega = self.quads[key]['state'][9:12]
        tau = np.array([self.quads[key]['L']*(self.quads[key]['m1'].thrust-self.quads[key]['m3'].thrust),
                        self.quads[key]['L']*(self.quads[key]['m2'].thrust-self.quads[key]['m4'].thrust),
                        0])
        omega_dot = np.dot(self.quads[key]['invI'], (tau - np.cross(omega, np.dot(self.quads[key]['I'],omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot

    def update(self):
            dt = self.dt
            key = 'q1'
            self.ode.set_initial_value(self.quads[key]['state'],0).set_f_params(key)
            self.quads[key]['state'] = self.ode.integrate(self.ode.t + dt)
            self.quads[key]['state'][6:9] = self.wrap_angle(self.quads[key]['state'][6:9])
            #self.quads[key]['state'][2] = np.clip(self.quads[key]['state'][2], self.range['linear_z'][0], self.range['linear_z'][1])
            self.quads[key]['state'][3] = 0
            self.quads[key]['state'][5] = np.clip(self.quads[key]['state'][5], self.range['linear_zrate'][0], self.range['linear_zrate'][1])
            self.quads[key]['state'][7] = 0
            self.quads[key]['state'][10] = 0

    def action(self,speeds):
        # speeds[0] = (speeds[1] + speeds[3])/2
        # speeds[2] = speeds[0]
        quad_name = 'q1'
        self.quads[quad_name]['m1'].set_speed(speeds[0])
        self.quads[quad_name]['m2'].set_speed(speeds[1])
        self.quads[quad_name]['m3'].set_speed(speeds[2])
        self.quads[quad_name]['m4'].set_speed(speeds[3])

    def get_position(self,quad_name):
        return self.quads[quad_name]['state'][0:3]

    def get_orientation(self,quad_name):
        return self.quads[quad_name]['state'][6:9]

    def get_state(self,quad_name):
        return self.quads[quad_name]['state']

    def get_motor(self,quad_name):
        speed = np.array([self.quads[quad_name]['m1'].speed,
                 self.quads[quad_name]['m2'].speed,
                 self.quads[quad_name]['m3'].speed,
                 self.quads[quad_name]['m4'].speed])
        return speed

    def set_position(self,quad_name,position):
        self.quads[quad_name]['state'][0:3] = position

    def set_orientation(self,quad_name,orientation):
        self.quads[quad_name]['state'][6:9] = orientation

    def reset_quads(self):
        key = 'q1'
        self.quads[key]['state'] = self.init[key]['state']
        ''''''
        self.reset_pointer %= len(self.resetstate)
        state = self.resetstate[self.reset_pointer]
        self.reset_pointer += 1
        state = self.dis2cont(dis_s=state)
        self.quads[key]['state'][0:3] = [0, 0, state[0]]
        self.quads[key]['state'][3:6] = [0, 0, state[1]]
        #self.quads[key]['state'][0:3] = [0, 0, uniform(self.range['linear_z'][0], self.range['linear_z'][1])]
        #self.quads[key]['state'][3:6] = [0, 0, uniform(self.range['linear_zrate'][0], self.range['linear_zrate'][1])]
        #self.quads[key]['state'][6:9] = [0, uniform(-0.001, 0.001), 0]
        #self.quads[key]['state'][9:] = [0, uniform(-0.001, 0.001), 0]
        self.quads[key]['m1'].speed = self.init[key]['m1_speed']
        self.quads[key]['m2'].speed = self.init[key]['m2_speed']
        self.quads[key]['m3'].speed = self.init[key]['m3_speed']
        self.quads[key]['m4'].speed = self.init[key]['m4_speed']
        # print("reset quadcopter")
        return self._state_cont2dis()

    def one_step(self, action, dt=0.05):
        self.dt = dt
        speed0 = self.action2motor(action)
        speeds = [np.array([5300]), speed0, np.array([5300]), speed0]
        s_cont = np.array([self.get_state('q1')[2], self.get_state('q1')[5]])
        self.action(speeds=speeds)
        self.update()
        s_, a = self._state_cont2dis(), self._motor2action()
        r, done, success = self.reward_ifdone(s_, a)
        s_cont_ = np.array([self.get_state('q1')[2], self.get_state('q1')[5]])
        a_cont_ = self._get_action(speed0)
        return s_, r, done, success, s_cont, s_cont_, a_cont_

    def reward_ifdone(self,s_, a):
        r = self.rewardscheme[tuple(np.hstack((s_, a)))]
        done, success = False, False
        if self.get_state('q1')[2] < self.range['linear_z'][0] or self.get_state('q1')[2] > self.range['linear_z'][1]:
            done = True
        if self.range['linear_z'][0] * 0.1 < self.get_state('q1')[2] < self.range['linear_z'][1] * 0.1:
            success = True
        return r, done, success

    def _state_cont2dis(self):
        state = self.get_state('q1')
        state = np.array([state[2], state[5]])
        t = np.array([0, 0])
        state = state - t
        state_num = np.array(self.state_action_num[0:-1])
        action_num = np.array(self.state_action_num[-1:])
        ds = np.zeros_like(state,dtype=int)
        for i, key in enumerate(self.range):
            margin = (self.range[key][1]-self.range[key][0])/state_num[i]
            ds[i] = int((state[i] - self.range[key][0])//margin)
            ds[i] = np.clip(ds[i], 0, self.state_action_num[i]-1)
        return ds

    def _motor2action(self):
        motor = self.get_motor('q1')
        motor = np.array(motor[1])
        ac = np.zeros_like(motor, dtype=int)
        for i, m in enumerate(motor):
            action_margin = (self.MOTOR_LIMITS[1] - self.MOTOR_LIMITS[0]) / self.state_action_num[-1]
            ac[i] = (m - self.MOTOR_LIMITS[0]) // action_margin
            ac[i] = np.clip(ac, 0, self.state_action_num[-1]-1)
        return ac

    def action2motor(self,action):
        action_margin = (self.MOTOR_LIMITS[1] - self.MOTOR_LIMITS[0]) / self.state_action_num[-1]
        speed = (2 * action) *action_margin + self.MOTOR_LIMITS[0]
        return speed

    def _get_action(self, motor):
        ac = np.zeros_like(motor, dtype=float)
        for i, m in enumerate(motor):
            ac[i] = (m - self.MOTOR_LIMITS[0]) / (self.MOTOR_LIMITS[1] - self.MOTOR_LIMITS[0])
        return ac

    def dis2cont(self, dis_s):
        t = np.array([0, 0])
        state_num = np.array(self.state_action_num[0:-1])
        state = np.zeros_like(state_num, dtype = float)
        for i, key in enumerate(self.range):
            margin = (self.range[key][1] - self.range[key][0])/(state_num[i])
            state[i] = self.range[key][0] + (dis_s[i] + 0.5) * margin
        state = state + t
        return state

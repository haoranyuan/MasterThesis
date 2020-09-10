import quad_env, gui, PID_controller_logging, state_action_value
import numpy as np
from numpy.random import uniform

def PID_Multi(QUAD_DYNAMICS_UPDATE=0.05,
             reward_flag='default',
             validation=0,
             episode=5000,
             render=False,
             verbose=False):

    Range = {
        'linear_z': [-3, 3],
        'linear_zrate': [-0.5, 0.5],
    }
    action_num = np.int32(2)
    state_num = np.array([21, 11], dtype=np.int32)
    state_action_num = np.hstack((state_num, action_num))
    target_state = np.array([10, 5], dtype=np.int32)

    terminal_state = []
    for s1 in range(state_action_num[0]):
        for s2 in range(state_action_num[1]):
            for a in range(state_action_num[2]):
                if s1 == 0 or s1 == state_action_num[0]-1 or \
                   s1 == target_state[0] and s2 == target_state[1]:
                    terminal_state.append([s1, s2])

    QUADCOPTER = {
        'q1': {'position': [0, 0, uniform(Range['linear_z'][0], Range['linear_z'][1])], 'orientation': [0, 0, 0],
               'L': 0.3,
               'r': 0.1, 'prop_size': [10, 4.5], 'weight': 1.2, 'range': Range, 'sa_num': state_action_num,
               'terminal': terminal_state, 'target_state': target_state}}
    # [70000, 70000](6.5), [70000, 35000](7.7), [70000, 10000](10.15), [10000, 70000](10.17) [10000, 10000](8.40)
    CONTROLLER_PARAMETERS = {'Motor_limits': [5200, 5500],
                             'Tilt_limits': [-10, 10],
                             'Yaw_Control_Limits': [-900, 900],
                             'Z_XY_offset': 0,
                             'Linear_PID': {'P': [300, 300, 70000], 'I': [0.04, 0.04, 0], 'D': [450, 450, 70000]},
                             'Linear_To_Angular_Scaler': [1, 1, 0],
                             'Yaw_Rate_Scaler': 0.18,
                             'Angular_PID': {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                             }

    ctrl1 = PID_controller_logging.PID_3D(params=CONTROLLER_PARAMETERS,
                                          quad_params=QUADCOPTER,
                                          ctrl_noise=False)
    return ctrl1.policy
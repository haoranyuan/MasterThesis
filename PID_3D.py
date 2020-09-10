import quad_env, gui, PID_controller_logging, state_action_value
import numpy as np
from numpy.random import uniform
from tqdm import tqdm

def PID_Multi(QUAD_DYNAMICS_UPDATE=0.05,
             reward_flag='default',
             validation=0,
             episode=5000,
             render=True,
             verbose=False,
             ctrl_noise=1.):

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
    # [20000, 0]() [5000, 10000]
    CONTROLLER_PARAMETERS = {'Motor_limits': [5200, 5500],
                             'Tilt_limits': [-10, 10],
                             'Yaw_Control_Limits': [-900, 900],
                             'Z_XY_offset': 0,
                             'Linear_PID': {'P': [300, 300, 70000], 'I': [0.04, 0.04, 0], 'D': [450, 450, 70000]},
                             'Linear_To_Angular_Scaler': [1, 1, 0],
                             'Yaw_Rate_Scaler': 0.18,
                             'Angular_PID': {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                             }

    quad = quad_env.Quadcopter(motorlimits= CONTROLLER_PARAMETERS['Motor_limits'],
                               quads=QUADCOPTER,
                               rewardflag=reward_flag)
    if render:
        gui_object = gui.GUI(quads=QUADCOPTER)
    ctrl1 = PID_controller_logging.PID_3D(params=CONTROLLER_PARAMETERS,
                                          quad_params=QUADCOPTER,
                                          ctrl_noise=ctrl_noise)
    #errorlog1 = rl_3d_agent.errorlog('e_'+reward_flag+'.csv')
    print('initialization finished, begin episodes.')
    success_rate = []
    success_count = 0
    MAX_EP_STEP = 200
    # Initialize demo dataset
    reset_times = 0
    for i in tqdm(range(episode)):
        quad.reset_pointer -= reset_times
        state = quad.reset_quads()
        success = 0
        done = False
        reset_times = 0
        for j in range(MAX_EP_STEP):
            if done and not success:
                state = quad.reset_quads()
                reset_times += 1
            action = ctrl1.policy(state)
            state_, r, done, sc, statecont, statecont_, actioncont_ = quad.one_step(action=action,
                                                dt=QUAD_DYNAMICS_UPDATE)
            ''''''
            # done = False
            if j == 0:
                ctrl1.statecont_log.append(statecont)
                ctrl1.actioncont_log.append(actioncont_)
                ctrl1.ep_r.append(-100)
            ctrl1.store_trans(state, action, r, state_, statecont, statecont_, actioncont_)
            # print(state, action, r, state_)
            if render:
                gui_object.quads['q1']['position'] = quad.get_position('q1')
                gui_object.quads['q1']['orientation'] = quad.get_orientation('q1')
                gui_object.update()
            # else:
            state = state_
        '''
        try:
            demodata = np.concatenate((demodata, np.hstack((ctrl1.statecont_log, ctrl1.actioncont_log))), axis=0)
        except Exception:
            demodata = np.hstack((ctrl1.statecont_log, ctrl1.actioncont_log))
        '''
        try:
            demodata = np.concatenate((demodata, np.hstack((ctrl1.statecont_log, ctrl1.actioncont_log, np.array(ctrl1.ep_r)[:, np.newaxis]))), axis=0)
        except Exception:
            demodata = np.hstack((ctrl1.statecont_log, ctrl1.actioncont_log, np.array(ctrl1.ep_r)[:, np.newaxis]))
        if sum(np.prod((ctrl1.ep_s_[-10:] == target_state), axis=1)) > 3:
            success = 1
        success_rate.append(success)
        if verbose:
            if i % 10 == 0:
                print('episode:', i)


        ctrl1.ep_r = []
        ctrl1.ep_a = []
        ctrl1.ep_s_ = []
        ctrl1.ep_s = []
        ctrl1.statecontlog = []
        ctrl1.statecont_log = []
        ctrl1.actioncont_log = []
    print('Averge reward per episode:', sum(demodata[:, 3])/episode)
    np.savetxt('demodata.csv', demodata, delimiter=',')

if __name__ == "__main__":
    PID_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=False, reward_flag='default', validation=0, episode=2000, ctrl_noise=0.8)

import quad_env,gui,rl_3d_agent,state_action_value
from tqdm import tqdm
import numpy as np
from numpy.random import uniform
from os import remove
from os.path import isfile
import os

def RL_Multi(QUAD_DYNAMICS_UPDATE=0.2,
             reward_flag='default',
             learningrate=0.5,
             validation=0,
             episode=3000,
             render=True,
             verbose=False,
             file_dir='results',
             drawpolicy=False,
             iteration=None,
             training_data=False,
             fuzzy_validation=False):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        print('RL: new file directory created at:', file_dir)
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
        'q1': {'position': [0, 0, uniform(Range['linear_z'][0], Range['linear_z'][1])], 'orientation': [0, 0, 0], 'L': 0.3,
               'r': 0.1, 'prop_size': [10, 4.5], 'weight': 1.2, 'range': Range, 'sa_num': state_action_num,
                'terminal': terminal_state, 'target_state': target_state}}

    CONTROLLER_PARAMETERS = {'Motor_limits': [5200, 5500]}
    sav = state_action_value.StateActionValue(state_action_num=state_action_num,
                                              reward_flag=reward_flag,
                                              file_dir=file_dir)
    quad = quad_env.Quadcopter(motorlimits=CONTROLLER_PARAMETERS['Motor_limits'],
                               quads=QUADCOPTER,
                               rewardflag=reward_flag,
                               file_dir=file_dir)
    if render:
        gui_object = gui.GUI(quads=QUADCOPTER)
    if validation and not fuzzy_validation:
        E = 1
        if fuzzy_validation:
            E = 0.9
    else:
        E = 0.7
    ctrl1 = rl_3d_agent.RL_3D(change_sav=sav.change_sav,
                              get_sav=sav.get_sav,
                              lr=learningrate,
                              params=QUADCOPTER,
                              epsilon=E)
    #errorlog1 = rl_3d_agent.errorlog('e_'+reward_flag+'.csv')
    print('RL: initialization finished, begin episodes.')

     # episodes
    success_rate = []
    success_log = []
    if isfile(file_dir+'sc_scatter_'+reward_flag+'val'+str(validation)+'.csv'):
        success_rate = list(np.genfromtxt(file_dir+'sc_scatter_'+reward_flag+'val'+str(validation)+'.csv', delimiter=','))
        print('load success rate data')

    sav_old = np.copy(sav.get_sav())
    #MAX_EP_STEP = 500
    MAX_EP_STEP = 200
    reset_times = 0
    for i in tqdm(range(episode)):
        #if not validation:
        quad.reset_pointer -= reset_times
        reset_times = 0
        state = quad.reset_quads()
        success = 0
        done = 0
        reward = 0
        lr_decay = 0.99999
        for j in range(MAX_EP_STEP):
            # If not success but cross the boarder then reset
            if done and not success:
                state = quad.reset_quads()
                reset_times += 1
            action = ctrl1.policy(state)
            state_, r, done, sc, statecont, statecont_, actioncont_ = quad.one_step(action=action,
                                            dt=QUAD_DYNAMICS_UPDATE)
            # done = False # always set done to False to prevent premature reset
            if done:
                r = -100
            success_log.append(sc)
            ''''''
            if j == 0:
                ctrl1.statecont_log.append(statecont)
                ctrl1.actioncont_log.append(actioncont_)
            ctrl1.store_trans(state, action, r, state_, statecont, statecont_, actioncont_)
            reward += r
            if not validation:
                ctrl1.learn()
                ctrl1.lr = max(1e-3, ctrl1.lr * lr_decay)
            if render:
                gui_object.quads['q1']['position'] = quad.get_position('q1')
                gui_object.quads['q1']['orientation'] = quad.get_orientation('q1')
                gui_object.update()



            # Keep the previous state at the boarder if reset
            state = state_
        if sum(np.array(success_log[-10:]) == 1) > 3:
            success = 1
        success_rate.append(np.array([success, reward]))
        if verbose:
            if i % 10 == 0:
                print('RL:episode={0:5d}, r_total={1:8.2f}, success:{2}, learning rate:{3}, progress:{4}'
                      .format(i, np.sum(ctrl1.ep_r), success, ctrl1.lr, np.linalg.norm(sav_old - sav.get_sav())))
            sav_old = np.copy(sav.get_sav())
        if validation or training_data:
            try:
                agentdata = np.concatenate((agentdata, np.hstack((ctrl1.statecont_log, ctrl1.actioncont_log))), axis=0)
            except Exception:
                agentdata = np.hstack((ctrl1.statecont_log, ctrl1.actioncont_log))
        ctrl1.ep_r = []
        ctrl1.ep_a = []
        ctrl1.ep_s_ = []
        ctrl1.ep_s = []
        ctrl1.statecontlog = []
        ctrl1.statecont_log = []
        ctrl1.actioncont_log = []
        success_log = []

    if not validation:
        sav.save_file()
    # Save continuous state log

    # Save success log
    np.savetxt(file_dir+'sc_scatter_'+reward_flag+'val'+str(validation)+'.csv', success_rate, delimiter=',')
    # Save trajectories
    if validation:
        if iteration is not None:
            name = file_dir + 'agentdata_'+reward_flag+str(iteration)+'.csv'
        else:
            name = file_dir + 'agentdata_' + reward_flag + '.csv'
        try:
            remove(name)
        except Exception:
            pass
        np.savetxt(name, agentdata, delimiter=',')
    if not validation and training_data:
        if iteration is not None:
            name = file_dir + 'agenttrain_'+reward_flag+str(iteration)+'.csv'
        else:
            name = file_dir + 'agenttrain_'+reward_flag+'.csv'
        np.savetxt(name, agentdata, delimiter=',')

    if drawpolicy:
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        policy = np.ones(shape=[21, 11])
        SAV = sav.get_sav()
        for ij, sa1 in enumerate(SAV):
            for j, sa in enumerate(sa1):
                a = np.argmax(sa)
                policy[ij, j] = a
        ax.imshow(policy, cmap='Blues')
        plt.show()

if __name__ == "__main__":
    for i in range(1):
        RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=True, reward_flag='A', validation=0, episode=2000, file_dir='RL_results/', verbose=False, drawpolicy=False, iteration=i, training_data=False)
        RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=False, reward_flag='A', validation=1, episode=231, file_dir='RL_results/', iteration=i)


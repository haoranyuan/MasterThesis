import numpy as np
from tqdm import tqdm
from AL_prior import reward_reconstruct
from rl_3D import RL_Multi
import os
import shutil
from policy_mixter import policy_mix
from policy_eval_recurrenct import policy_eval
from control_evaluate import control_eval

# demo_file = 'AL_results/p_projection_good_07noise/demodata.csv'
# DIR_ = 'AL_results/p_projection_good/'
DIR_ = 'AL_results/p_projection_good_08noise/'
REPEAT = 10
ITER = 20
for ij in tqdm(range(REPEAT)):
    ii = ij + 0
    DIR = DIR_ + 'p_projection_good_08noise'+str(ii) +'/'
    for iter in range(ITER):
        print('AL: iteration:', iter)
        if os.path.isfile(DIR + 'iter' + str(iter) + '/' + 'reward_AL.csv'):
            print('AL: reward saved at: ', DIR + 'iter' + str(iter) + '/')
        RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, reward_flag='AL', validation=1, render=False, episode=231,
                    file_dir=DIR + 'iter' + str(iter) + '/', fuzzy_validation=False)
        if os.path.isfile(DIR + 'iter' + str(iter) + '/' + 'agentdata_AL.csv'):
            print('AL: agent traj saved at:', DIR + 'iter' + str(iter) + '/')
    # ------------------------------------------
    policy_mix(DIR, PRIOR=True, EP_number=ITER+1, PLOT=False)
    policy_eval(DIR, iter=ITER, PLOT=False, force_evaluate=True)
    control_eval(DIR, iter=ITER, force_evaluate=True)
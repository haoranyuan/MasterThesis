from rl_3D import RL_Multi
import numpy as np
import os
import shutil
from tqdm import tqdm

DIR = 'RL_results/'
ITER = 20
REPEATS = 10
for ii in tqdm(range(REPEATS)):
    reward_file = DIR + 'reward_B.csv'
    DIR_ = DIR + 'RL' + str(ii) + '/'
    os.mkdir(DIR + 'RL' + str(ii))
    _ = shutil.copy(reward_file, DIR_)
    for i in range(ITER):
        RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=False, reward_flag='A', validation=0, episode=1231,
                 file_dir=DIR_, verbose=False, drawpolicy=False, iteration=i, training_data=False)
        RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=False, reward_flag='A', validation=1, episode=231,
                 file_dir=DIR_, iteration=i)
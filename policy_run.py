from rl_3D import RL_Multi as RL_sav
from rl_policy_dir import RL_Multi as RL_policy

DIR = 'policies/bad/'
RL_sav(QUAD_DYNAMICS_UPDATE=0.2, render=False, reward_flag='AL', validation=1, episode=231, file_dir=DIR, iteration=None)
RL_policy(QUAD_DYNAMICS_UPDATE=0.2, render=False, validation=1, episode=231, file_dir=DIR,
             policy_dir=DIR + 'mixed_policy.csv', fuzzy=False)

DIR = 'policies/good/'
RL_sav(QUAD_DYNAMICS_UPDATE=0.2, render=False, reward_flag='AL', validation=1, episode=231, file_dir=DIR, iteration=None)
RL_policy(QUAD_DYNAMICS_UPDATE=0.2, render=False, validation=1, episode=231, file_dir=DIR,
             policy_dir=DIR + 'mixed_policy.csv', fuzzy=False)

DIR = 'policies/p_bad/'
RL_sav(QUAD_DYNAMICS_UPDATE=0.2, render=False, reward_flag='AL', validation=1, episode=231, file_dir=DIR, iteration=None)
RL_policy(QUAD_DYNAMICS_UPDATE=0.2, render=False, validation=1, episode=231, file_dir=DIR,
             policy_dir=DIR + 'mixed_policy.csv', fuzzy=False)

DIR = 'policies/p_good/'
RL_sav(QUAD_DYNAMICS_UPDATE=0.2, render=False, reward_flag='AL', validation=1, episode=231, file_dir=DIR, iteration=None)
RL_policy(QUAD_DYNAMICS_UPDATE=0.2, render=False, validation=1, episode=231, file_dir=DIR,
             policy_dir=DIR + 'mixed_policy.csv', fuzzy=False)



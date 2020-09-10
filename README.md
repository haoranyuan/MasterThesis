# Python code for Master Thesis
## by Haoran Yuan
[Full text here]

[Full text here]: https://repository.tudelft.nl/islandora/object/uuid%3Afd8ab65d-7869-4b21-b09d-bdba9a74cb36
### Files
1. AL_prior.py, AL_prior_copy.py: SAL agent.
2. AL_quadrotor.py: normal AL agent.
3. Another_solver.py: Projection method.
4. control_evaluate.py: evaluation of agent's performance.
5. experiments.py, experiments_copy.py: AL expirements.
6. experiments_PID.py: RL experiments without reward functions from the AL algorithm.
7. experiments_sal.py, experiments_sal_copy.py: SAL experiments.
8. gui.py: GUI for 3D animation.
9. PID_3D.py: Simulation on PID controllers.
10. PID_controller_logging.py: PID controller.
11. plot_AL_results.py, plot_SAL_AL.py, plot_SAL_results.py, plot_timetraces.py: Plotting results.
12. plot_policies.py: Plotting policies in 2D heat map.
13. policy_eval_recurrent.py, policy_eval_recurrent.py: Evaluation of the policies produced by SAL and AL algorithms.
14. policy_evaluate.py: Plotting the similarity curve of the target policy and the produced policies.
15. policy_mixer.py: Mixing historic policies into new policies for AL or SAL.
16. quad_env.py: Quadrotor environment for simulations.
17. rewardconstruct.py: Reward function contstructions.
18. rl_3D.py: RL simulation with Q-learning.
19. rl_3d_agent.py: Q-learning agent.
20. rl_policy_dir.py: The simulation that directly use policies rather than state-action value functions.
21. state_action_value.py: State-action value function.

### How to use
1. Run RL (Q-learning) experiments: Open rl_3D.py, scroll down to the last part. Under "if __name__ == "__main__"" change parameters then run.
2. Run RL (Q-learning) with policy files: Policy files contains numbers in [0, 1] as the probability of selecting "action 1". Open rl_policy_dir.py, scroll down.
Choose the directory that contains policy file then run.
3. Run AL and SAL experiments: Open experiments.py or experiments_sal.py, enter directories and other parameters, then run.
4. Plot AL and SAL results: Open plot_AL_results.py, plot_SAL_AL.py, plot_SAL_results.py or plot_timetraces.py, select directories, then run.

### Notice
1. The default setting for number of repeats of AL and SAL is 10. It usually takes more than 4 hours to finish.
2. Each folder of results is more than 3 GB.

import os, shlex, subprocess
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

commands = []

#1-1
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --unsupervised_exploration --exp_name q1_env1_random"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --exp_name q1_env2_random"
# commands.append(command)
#1-2
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --unsupervised_exploration --exp_name q1_hard_rnd"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_boltzmann --unsupervised_exploration --exp_name q1_alg_med"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_boltzmann --unsupervised_exploration --exp_name q1_alg_hard"
# commands.append(command)
# #2-1
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift 1 --exploit_rew_scale 100"
# commands.append(command)
# #2-2
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_5000 --exploit_rew_shift 1 --exploit_rew_scale 100"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_15000 --exploit_rew_shift 1 --exploit_rew_scale 100"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_5000"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_15000"
# commands.append(command)
# #2-3
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.02 --exp_name q2_alpha0.02 --exploit_rew_shift 1 --exploit_rew_scale 100"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.5 --exp_name q2_alpha0.5 --exploit_rew_shift 1 --exploit_rew_scale 100"
# commands.append(command)
#3-1
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_medium_dqn"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_medium_cql --exploit_rew_shift 1 --exploit_rew_scale 100"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_hard_dqn"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_hard_cql --exploit_rew_shift 1 --exploit_rew_scale 100"
# commands.append(command)


if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
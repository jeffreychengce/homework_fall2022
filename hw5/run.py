import os, shlex, subprocess
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

commands = []
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --unsupervised_exploration --exp_name q1_env1_random"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd"
# commands.append(command)
# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --exp_name q1_env2_random"
# commands.append(command)

# command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --unsupervised_exploration --exp_name q1_hard_rnd"
# commands.append(command)
command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_boltzmann --unsupervised_exploration --exp_name q1_alg_med"
commands.append(command)
command = "python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_boltzmann --unsupervised_exploration --exp_name q1_alg_hard"
commands.append(command)



if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
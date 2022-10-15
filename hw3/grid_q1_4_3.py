import shlex, subprocess

commands = []

commands.append("python ./cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --eval_batch_size 100 --exp_name q3_hparam1")
commands.append("python ./cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --eval_batch_size 5000 --exp_name q3_hparam2")
commands.append("python ./cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --eval_batch_size 20000 --exp_name q3_hparam3")


if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
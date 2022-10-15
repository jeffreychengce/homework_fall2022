import shlex, subprocess

commands = []
lr_list = [1, 2, 3]

for lr in lr_list:
    #command = "python ./cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_{lr} --seed {lr}".format(lr=lr)
    #commands.append(command)
    command = "python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_{lr} --double_q --seed {lr}".format(lr=lr)
    commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
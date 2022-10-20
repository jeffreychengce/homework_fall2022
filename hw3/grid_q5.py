import shlex, subprocess

commands = []

commands.append("python cs285/scripts/run_hw3_sac.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.99 --scalar_log_freq 1500 -n 200000 -l 2 -s 256 -b 1500 -eb 1500 -lr 0.00003 --init_temperature 0.1 --exp_name q6b_sac_HalfCheetah_lr_3e-5_tb_1500_actorfreq_10 --seed 1 -tb 1500 --actor_update_frequency 10")
commands.append("python cs285/scripts/run_hw3_sac.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.99 --scalar_log_freq 1500 -n 200000 -l 2 -s 256 -b 1500 -eb 1500 -lr 0.00003 --init_temperature 0.1 --exp_name q6b_sac_HalfCheetah_lr_3e-5_tb_1500_actorfreq_10_criticfreq_10 --seed 1 -tb 1500 --actor_update_frequency 10 --critic_target_update_frequency 10")


if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
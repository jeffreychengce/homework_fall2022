Script for plotting available upon request.

## Question 1: DQN MsPacman

The following command was run to produce the MsPacman learning curve:

```
python python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q1
```

## Question 2: DQN and DDQN LunarLander

The following command were run to produce the LunarLander learning curves:
```
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v2 \ 
    --exp_name q2_dqn_1 --seed 1
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v2 \ 
    --exp_name q2_dqn_2 --seed 2
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v2 \ 
    --exp_name q2_dqn_3 --seed 3

python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v2 \ 
    --exp_name q2_doubledqn_1 --double_q --seed 1
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v2 \ 
    --exp_name q2_doubledqn_2 --double_q --seed 2
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v2 \ 
    --exp_name q2_doubledqn_3 --double_q --seed 3
```

## Question 3: LunarLander Hyperparameter Experiments

The following commands were run for the LunarLander environment:
```
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v2 \ 
    --exp_name q2_dqn_1 --seed 1

python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \ 
    --batch_size 16 --exp_name q3_hparam1
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \ 
    --batch_size 128 --exp_name q3_hparam2
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \ 
    --batch_size 2048 --exp_name q3_hparam3
```

## Question 4: CartPole Actor-Critic

The following commands were run for the CartPole environment:

```
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 \ 
    -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 \ 
    -n 100 -b 1000 --exp_name q4_ac_1_100 -ntu 1 -ngsptu 100
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 \ 
    -n 100 -b 1000 --exp_name q4_ac_100_1 -ntu 100 -ngsptu 1
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 \ 
    -n 100 -b 1000 --exp_name q4_ac_10_10 -ntu 10 -ngsptu 10
```

# Question 5: Actor-Critic

The following commands were run for the InvertedPendulum and HalfCheetah environments:

```
python cs285/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v4 \ 
    --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 \ 
    --exp_name q5_10_10 -ntu 10 -ngsptu 10

python cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v4 \ 
    --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 \ 
    -b 30000 -eb 1500 -lr 0.02 --exp_name q5_10_10 -ntu 10 -ngsptu 10
```

# Question 5: Soft Actor-Critic

The following commands were run for the InvertedPendulum and HalfCheetah environments:

```
python cs285/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v4 \ 
    --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 \ 
    --exp_name q5_10_10 -ntu 10 -ngsptu 10

python cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v4 \ 
    --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 \ 
    -b 30000 -eb 1500 -lr 0.02 --exp_name q5_10_10 -ntu 10 -ngsptu 10
```
## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions. There are two new package requirements (`opencv-python` and `gym[atari]`) beyond what was used in the previous assignments; make sure to install these with `pip install -r requirements.txt`, `pip install gym[atari]`, and `pip install gym[accept-rom-license]` 
if you are running the assignment locally.

2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badges below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2022/blob/main/hw3/cs285/scripts/run_hw3_dqn.ipynb) **Part I (Q-learning)** 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2022/blob/main/hw3/cs285/scripts/run_hw3_actor_critic.ipynb)     **Part II (Actor-critic)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2022/blob/main/hw3/cs285/scripts/run_hw3_soft_actor_critic.ipynb)     **Part III (Soft Actor-critic)** 

## Complete the code

The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with `TODO: get this from hw1 or hw2`.

- [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
- [infrastructure/utils.py](cs285/infrastructure/utils.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy.py)

You will then need to implement new routines in the following files for homework 3 part 1 (Q-learning):
- [agents/dqn_agent.py](cs285/agents/dqn_agent.py)
- [critics/dqn_critic.py](cs285/critics/dqn_critic.py)
- [policies/argmax_policy.py](cs285/policies/argmax_policy.py)

and in the following files for part 2 (actor-critic):
- [agents/ac_agent.py](cs285/agents/ac_agent.py)
- [critics/bootstrapped_continuous_critic.py](cs285/critics/bootstrapped_continuous_critic.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy.py)

The relevant sections are marked with `TODO`.

You may also want to look through [scripts/run_hw3_dqn.py](cs285/scripts/run_hw3_dqn.py) and [scripts/run_hw3_actor_critic](cs285/scripts/run_hw3_actor_critic.py) (if running locally) or [scripts/run_hw3.ipynb](cs285/scripts/run_hw3.ipynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

See the [assignment PDF](cs285_hw3.pdf) for more details on what files to edit.

1-1: 9
100-1: 200 in 40, drop at 70
1-100: 200 in 40
10-10: 200 in 20, drop at 80

python cs285/scripts/run_hw3_sac.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.99 --scalar_log_freq 1000 -n 100000 -l 2 -s 256 -b 1000 -eb 2000 -lr 0.0003 --init_temperature 0.1 --exp_name q6a_sac_InvertedPendulum --seed 1 

python cs285/scripts/run_hw3_sac.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.99 --scalar_log_freq 1500 -n 200000 -l 2 -s 256 -b 1500 -eb 1500 -lr 0.0003 --init_temperature 0.1 --exp_name q6b_sac_HalfCheetah --seed 1

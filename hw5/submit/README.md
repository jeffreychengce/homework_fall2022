Script for plotting available upon request.

## Question 1: 

The following commands were run:

```
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 \ 
    --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 \ 
    --unsupervised_exploration --exp_name q1_env1_random
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --unsupervised_exploration --exp_name q1_env2_random

python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 \ 
    --use_rnd --unsupervised_exploration --exp_name q1_hard_rnd
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --use_boltzmann --unsupervised_exploration --exp_name q1_alg_med
python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 \ 
    --use_boltzmann --unsupervised_exploration --exp_name q1_alg_hard
```

## Question 2: 

The following commands were run:
```
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --exp_name q2_dqn --use_rnd --unsupervised_exploration \ 
    --offline_exploitation --cql_alpha=0
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --exp_name q2_cql --use_rnd --unsupervised_exploration \ 
    --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift 1 \
    --exploit_rew_scale 100

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --use_rnd --num_exploration_steps=5000 --offline_exploitation \ 
    --cql_alpha=0.1 --unsupervised_exploration \ 
    --exp_name q2_cql_numsteps_5000 --exploit_rew_shift 1 \ 
    --exploit_rew_scale 100
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --use_rnd --num_exploration_steps=15000 --offline_exploitation \ 
    --cql_alpha=0.1 --unsupervised_exploration \ 
    --exp_name q2_cql_numsteps_15000 --exploit_rew_shift 1 \ 
    --exploit_rew_scale 100
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --use_rnd --num_exploration_steps=5000 --offline_exploitation \ 
    --cql_alpha=0.0 --unsupervised_exploration \ 
    --exp_name q2_dqn_numsteps_5000
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --use_rnd --num_exploration_steps=15000 --offline_exploitation \ 
    --cql_alpha=0.0 --unsupervised_exploration \
    --exp_name q2_dqn_numsteps_15000
    
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --use_rnd --unsupervised_exploration --offline_exploitation \ 
    --cql_alpha=0.02 --exp_name q2_alpha0.02 --exploit_rew_shift 1 \ 
    --exploit_rew_scale 100
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --use_rnd --unsupervised_exploration --offline_exploitation \ 
    --cql_alpha=0.5 --exp_name q2_alpha0.5 --exploit_rew_shift 1 \ 
    --exploit_rew_scale 100

```

## Question 3: 

The following commands were run:
```
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 \ 
    --exp_name q3_medium_dqn
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \ 
    --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 \ 
    --exp_name q3_medium_cql --exploit_rew_shift 1 \
    --exploit_rew_scale 100
python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 \ 
    --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 \ 
    --exp_name q3_hard_dqn
python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 \ 
    --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 \ 
    --exp_name q3_hard_cql --exploit_rew_shift 1 \ 
    --exploit_rew_scale 100
```

## Question 4: 

The following commands were run:

```
python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \ 
    --use_rnd --num_exploration_steps=20000 --unsupervised_exploration \ 
    --awac_lambda={l} --exp_name q4_awac_easy_unsupervised_lam{l}
python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \ 
    --use_rnd --num_exploration_steps=20000 --unsupervised_exploration \ 
    --awac_lambda={l} --exp_name q4_awac_medium_unsupervised_lam{l}
python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
    --use_rnd --num_exploration_steps=20000 --awac_lambda={l} \ 
    --exp_name q4_awac_easy_supervised_lam{l}
python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \ 
    --use_rnd --num_exploration_steps=20000 --awac_lambda={l} \ 
    --exp_name q4_awac_medium_supervised_lam{l}
]
```
l = [0.1,1,2,10,20,50]

# Question 5: 

The following commands were run:

```
python cs285/scripts/run_hw5_iql.py --no_gpu --env_name PointmassEasy-v0 \ 
    --exp_name q5_iql_easy_supervised_lam{l}_tau{t} --use_rnd \ 
    --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={t}
python cs285/scripts/run_hw5_iql.py --no_gpu --env_name PointmassEasy-v0 \
    --exp_name q5_iql_easy_unsupervised_lam{l}_tau{t} \
    --unsupervised_exploration --use_rnd --num_exploration_steps=20000 \
    --awac_lambda={l} --iql_expectile={t}
python cs285/scripts/run_hw5_iql.py --no_gpu --env_name PointmassMedium-v0 \ 
    --exp_name q5_iql_medium_supervised_lam{l}_tau{t} --use_rnd \ 
    --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={t}
python cs285/scripts/run_hw5_iql.py --no_gpu --env_name PointmassMedium-v0 \ 
    --exp_name q5_iql_medium_unsupervised_lam{l}_tau{t} \ 
    --unsupervised_exploration --use_rnd --num_exploration_steps=20000 \ 
    --awac_lambda={l} --iql_expectile={t}
```
l = [1, 2, 20, 1] # easy-sup, easy-unsup, medium-sup, medium-unsup
tau = [0.5, 0.7, 0.9, 0.99] #[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
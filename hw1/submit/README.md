## Section 1 Question 2: Behavioral Cloning

For the Ant environment results, run:
```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1 --ep_len 1000 --eval_batch_size 5000
```

For the Hopper environment results, run: 
```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Hopper.pkl \
    --env_name Hopper-v4 --exp_name bc_hopper --n_iter 1 \
    --expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
    --video_log_freq -1 --ep_len 1000 --eval_batch_size 5000
```

## Section 1 Question 3: Hyperparameter

The number of hidden layers is varied using the following commands:
```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1 --ep_len 1000 --eval_batch_size 5000 \
    --n_layers 2
```
Then ```n_layers``` is varied from 2 to 64 by powers of 2.



## Section 2 Question 2: DAgger
For the Ant environment results, run:
```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1 --ep_len 1000 --eval_batch_size 5000
```

For the Hopper environment results, run: 
```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Hopper.pkl \
    --env_name Hopper-v4 --exp_name dagger_hopper --n_iter 1 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
    --video_log_freq -1 --ep_len 1000 --eval_batch_size 5000
```

Script for plotting 1-3 and 2-2 available upon request.
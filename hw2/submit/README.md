## Experiment 1: CartPole

The following commands were run to produce the CartPole learning curves:

```
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-dsa --exp_name q1_sb_no_rtg_dsa

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg -dsa --exp_name q1_sb_rtg_dsa

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg --exp_name q1_sb_rtg_na

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
-dsa --exp_name q1_lb_no_rtg_dsa

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
-rtg -dsa --exp_name q1_lb_rtg_dsa

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
-rtg --exp_name q1_lb_rtg_na

```

## Experiment 2: InvertedPendulum

The following command was run for the batch size and learning rate needed to reach the maximum score in less than 100 iterations: 
```
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 2000 -lr 5e-2 -rtg \
--exp_name q2_b2000_r5e-2

```

## Experiment 3: LunarLander

The following command was run for the LunarLander environment:
```
python cs285/scripts/run_hw2.py \
--env_name LunarLanderContinuous-v2 --ep_len 1000 \
--discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 \
--reward_to_go --nn_baseline --exp_name q3_b40000_r0.005
```

## Experiment 4: HalfCheetah
b10000lr0.005: 
b10000lr0.01: 
b10000lr0.02: 
b30000lr0.005: 
b30000lr0.01:
b30000lr0.02:
b50000lr0.005: 
b50000lr0.01:
b50000lr0.02:

# Experiment 5: HopperV4


Script for plotting available upon request.
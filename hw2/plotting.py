# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:28:22 2022

@author: Jeffrey
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import glob
#%%
def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

#%%

# q1
q1_sb_no_rtg_dsa_CartPole = glob.glob('**/q2_pg_q1_sb_no_rtg_dsa_CartPole-v0_25-09-2022_12-02-11/*')[0]
q1_sb_rtg_dsa_CartPole = glob.glob('**/q2_pg_q1_sb_rtg_dsa_CartPole-v0_25-09-2022_12-05-14/*')[0]
q1_sb_rtg_na_CartPole = glob.glob('**/q2_pg_q1_sb_rtg_na_CartPole-v0_25-09-2022_18-50-46/*')[0]

q1_lb_no_rtg_dsa_CartPole = glob.glob('**/q2_pg_q1_lb_no_rtg_dsa_CartPole-v0_25-09-2022_12-11-17/*')[0]
q1_lb_rtg_dsa_CartPole = glob.glob('**/q2_pg_q1_lb_rtg_dsa_CartPole-v0_25-09-2022_12-22-21/*')[0]
q1_lb_rtg_na_CartPole = glob.glob('**/q2_pg_q1_lb_rtg_na_CartPole-v0_25-09-2022_18-54-25/*')[0]

_, q1_sb_no_rtg_dsa_CartPole_returns = get_section_results(q1_sb_no_rtg_dsa_CartPole)
_, q1_sb_rtg_dsa_CartPole_returns = get_section_results(q1_sb_rtg_dsa_CartPole)
_, q1_sb_rtg_na_CartPole_returns = get_section_results(q1_sb_rtg_na_CartPole)

_, q1_lb_no_rtg_dsa_CartPole_returns = get_section_results(q1_lb_no_rtg_dsa_CartPole)
_, q1_lb_rtg_dsa_CartPole_returns = get_section_results(q1_lb_rtg_dsa_CartPole)
_, q1_lb_rtg_na_CartPole_returns = get_section_results(q1_lb_rtg_na_CartPole)

itr = np.arange(100)

fig1 = plt.figure()
plt.hlines(200,0,100,'r',label='Maximum Reward')
plt.plot(itr, q1_sb_no_rtg_dsa_CartPole_returns, label='No Reward to Go, No Standardized Advantages')
plt.plot(itr, q1_sb_rtg_dsa_CartPole_returns, label='Reward to Go, No Standardized Advantages')
plt.plot(itr, q1_sb_rtg_na_CartPole_returns, label='Reward to Go, Standardized Advantages')
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('CartPole Small Batch Policy Gradient Learning Curves')
plt.legend(loc='lower right')
fig1.set_size_inches(10, 6)
plt.show()

fig2 = plt.figure()
plt.hlines(200,0,100,'r',label='Maximum Reward')
plt.plot(itr, q1_lb_no_rtg_dsa_CartPole_returns, label='No Reward to Go, No Standardized Advantages')
plt.plot(itr, q1_lb_rtg_dsa_CartPole_returns, label='Reward to Go, No Standardized Advantages')
plt.plot(itr, q1_lb_rtg_na_CartPole_returns, label='Reward to Go, Standardized Advantages')
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('CartPole Large Batch Policy Gradient Learning Curves')
plt.legend(loc='lower right')
fig2.set_size_inches(10, 6)
plt.show()

#q2
q2_b2000_r7e_3_InvertedPendulum = glob.glob('**/q2_pg_q2_b2000_r7e-3_InvertedPendulum-v4_26-09-2022_01-51-12/*')[0]
_, q2_b2000_r7e_3_InvertedPendulum_returns = get_section_results(q2_b2000_r7e_3_InvertedPendulum)

fig4 = plt.figure()
plt.hlines(1000,0,100,'r',label='Maximum Reward')
plt.plot(itr, q2_b2000_r7e_3_InvertedPendulum_returns, label='batch=2000, lr=0.003')
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Inverted Pendulum Policy Gradient Learning Curve')
plt.legend(loc='lower right')
fig4.set_size_inches(10, 6)
plt.show()

#q3





#q4

q4_search_b10000_lr0_005 = glob.glob('**/q2_pg_q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_19-58-16/*')[0]
q4_search_b10000_lr0_01 = glob.glob('**/q2_pg_q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_19-58-17/*')[0]
q4_search_b10000_lr0_02 = glob.glob('**/q2_pg_q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_19-58-17/*')[0]

q4_search_b30000_lr0_005 = glob.glob('**/q2_pg_q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_19-58-17/*')[0]
q4_search_b30000_lr0_01 = glob.glob('**/q2_pg_q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_19-58-17/*')[0]
q4_search_b30000_lr0_02 = glob.glob('**/q2_pg_q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_19-58-17/*')[0]

q4_search_b50000_lr0_005 = glob.glob('**/q2_pg_q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_19-58-17/*')[0]
q4_search_b50000_lr0_01 = glob.glob('**/q2_pg_q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_19-58-17/*')[0]
q4_search_b50000_lr0_02 = glob.glob('**/q2_pg_q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_19-58-17/*')[0]

_, q4_search_b10000_lr0_01_returns = get_section_results(q4_search_b10000_lr0_01)
_, q4_search_b10000_lr0_02_returns = get_section_results(q4_search_b10000_lr0_02)
_, q4_search_b10000_lr0_005_returns = get_section_results(q4_search_b10000_lr0_005)

_, q4_search_b30000_lr0_01_returns = get_section_results(q4_search_b30000_lr0_01)
_, q4_search_b30000_lr0_02_returns = get_section_results(q4_search_b30000_lr0_02)
_, q4_search_b30000_lr0_005_returns = get_section_results(q4_search_b30000_lr0_005)

_, q4_search_b50000_lr0_01_returns = get_section_results(q4_search_b50000_lr0_01)
_, q4_search_b50000_lr0_02_returns = get_section_results(q4_search_b50000_lr0_02)
_, q4_search_b50000_lr0_005_returns = get_section_results(q4_search_b50000_lr0_005)

fig6 = plt.figure()

plt.plot(itr, q4_search_b10000_lr0_005_returns, label='b=10000, r=0.05')
plt.plot(itr, q4_search_b10000_lr0_01_returns, label='b=10000, r=0.01')
plt.plot(itr, q4_search_b10000_lr0_02_returns, label='b=10000, r=0.02')
plt.plot(itr, q4_search_b30000_lr0_005_returns, label='b=30000, r=0.05')
plt.plot(itr, q4_search_b30000_lr0_01_returns, label='b=30000, r=0.01')
plt.plot(itr, q4_search_b30000_lr0_02_returns, label='b=30000, r=0.02')
plt.plot(itr, q4_search_b50000_lr0_005_returns, label='b=50000, r=0.05')
plt.plot(itr, q4_search_b50000_lr0_01_returns, label='b=50000, r=0.01')
plt.plot(itr, q4_search_b50000_lr0_02_returns, label='b=50000, r=0.02')

plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Half Cheetah Policy Gradient Learning Curves')
plt.legend(loc='lower right')
fig6.set_size_inches(10, 6)
plt.show()





#q5

q5_b2000_r0_001_lambda1 = glob.glob('**/q2_pg_q5_b2000_r0.001_lambda1_Hopper-v4_26-09-2022_01-50-45/*')[0]
q5_b2000_r0_001_lambda99 = glob.glob('**/q2_pg_q5_b2000_r0.001_lambda0.99_Hopper-v4_26-09-2022_01-50-45/*')[0]
q5_b2000_r0_001_lambda98 = glob.glob('**/q2_pg_q5_b2000_r0.001_lambda0.98_Hopper-v4_26-09-2022_01-50-45/*')[0]
q5_b2000_r0_001_lambda95 = glob.glob('**/q2_pg_q5_b2000_r0.001_lambda0.95_Hopper-v4_26-09-2022_01-50-45/*')[0]
q5_b2000_r0_001_lambda0 = glob.glob('**/q2_pg_q5_b2000_r0.001_lambda0_Hopper-v4_26-09-2022_01-50-45/*')[0]

_, q5_b2000_r0_001_lambda1_returns = get_section_results(q5_b2000_r0_001_lambda1)
_, q5_b2000_r0_001_lambda99_returns = get_section_results(q5_b2000_r0_001_lambda99)
_, q5_b2000_r0_001_lambda98_returns = get_section_results(q5_b2000_r0_001_lambda98)
_, q5_b2000_r0_001_lambda95_returns = get_section_results(q5_b2000_r0_001_lambda95)
_, q5_b2000_r0_001_lambda0_returns = get_section_results(q5_b2000_r0_001_lambda0)

itr = np.arange(300)

fig7 = plt.figure()

plt.plot(itr, q5_b2000_r0_001_lambda1_returns, label='lambda=1')
plt.plot(itr, q5_b2000_r0_001_lambda99_returns, label='lambda=99')
plt.plot(itr, q5_b2000_r0_001_lambda98_returns, label='lambda=98')
plt.plot(itr, q5_b2000_r0_001_lambda95_returns, label='lambda=95')
plt.plot(itr, q5_b2000_r0_001_lambda0_returns, label='lambda=0')



plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Hopper Policy Gradient Learning Curves with Varying GAE Lambda')
plt.legend(loc='lower right')
fig7.set_size_inches(10, 6)
plt.show()














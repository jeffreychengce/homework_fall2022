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
q1_sb_rtg_na_CartPole = glob.glob('**/q2_pg_q1_sb_rtg_na_CartPole-v0_25-09-2022_12-08-05/*')[0]

q1_lb_no_rtg_dsa_CartPole = glob.glob('**/q2_pg_q1_lb_no_rtg_dsa_CartPole-v0_25-09-2022_12-11-17/*')[0]
q1_lb_rtg_dsa_CartPole = glob.glob('**/q2_pg_q1_lb_rtg_dsa_CartPole-v0_25-09-2022_12-22-21/*')[0]
q1_lb_rtg_na_CartPole = glob.glob('**/q2_pg_q1_lb_rtg_na_CartPole-v0_25-09-2022_12-32-23/*')[0]

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
q2_b2000_r5e_2_InvertedPendulum = glob.glob('**/q2_pg_q2_b2000_r5e-2_InvertedPendulum-v4_24-09-2022_01-48-49/*')[0]
_, q2_b2000_r5e_2_InvertedPendulum_returns = get_section_results(q2_b2000_r5e_2_InvertedPendulum)

fig4 = plt.figure()
plt.hlines(1000,0,100,'r',label='Maximum Reward')
plt.plot(itr, q2_b2000_r5e_2_InvertedPendulum_returns, label='batch=2000, lr=0.05')
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Inverted Pendulum Policy Gradient Learning Curve')
plt.legend(loc='lower right')
fig4.set_size_inches(10, 6)
plt.show()

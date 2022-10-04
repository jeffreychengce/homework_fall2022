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


#%% q1
q1_sb_no_rtg_dsa_CartPole = glob.glob('**/q2_pg_q1_sb_no_rtg_dsa_CartPole-v0_24-09-2022_00-35-05/*')[0]
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

#%% q2
q2_b2000_r7e_3_InvertedPendulum = glob.glob('**/q2_pg_q2_b2000_r7e-3_InvertedPendulum-v4_26-09-2022_01-51-12/*')[0]
_, q2_b2000_r7e_3_InvertedPendulum_returns = get_section_results(q2_b2000_r7e_3_InvertedPendulum)

fig3 = plt.figure()
plt.hlines(1000,0,100,'r',label='Maximum Reward')
plt.plot(itr, q2_b2000_r7e_3_InvertedPendulum_returns, label='batch=2000, lr=0.003')
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Inverted Pendulum Policy Gradient Learning Curve')
plt.legend(loc='lower right')
fig3.set_size_inches(10, 6)
plt.show()

#%% q3
q3_b40000_r0_005_LunarLanderContinuous= glob.glob('**/q2_pg_q3_b40000_r0.005_LunarLanderContinuous-v2_26-09-2022_02-48-31/*')[0]
_, q3_b40000_r0_005_LunarLanderContinuous_returns = get_section_results(q3_b40000_r0_005_LunarLanderContinuous)

fig4 = plt.figure()
#plt.hlines(1000,0,100,'r',label='Maximum Reward')
plt.plot(itr, q3_b40000_r0_005_LunarLanderContinuous_returns)
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('LunarLander Policy Gradient with Neural Network Baseline Learning Curve')
fig4.set_size_inches(10, 6)
plt.show()




#%% q4

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

fig5 = plt.figure()

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
plt.title('Half Cheetah Policy Gradient with Baseline Learning Curves')
plt.legend(loc='upper left')
fig5.set_size_inches(10, 6)
plt.show()

q4_b50000_r0_02 = glob.glob('**/q2_pg_q4_b50000_r0.02_HalfCheetah-v4_26-09-2022_02-52-42/*')[0]
q4_b50000_r0_02_rtg = glob.glob('**/q2_pg_q4_b50000_r0.02_rtg_HalfCheetah-v4_26-09-2022_02-52-42/*')[0]
q4_b50000_r0_02_nnbaseline = glob.glob('**/q2_pg_q4_b50000_r0.02_nnbaseline_HalfCheetah-v4_26-09-2022_02-52-42/*')[0]
q4_b50000_r0_02_rtg_nnbaseline = glob.glob('**/q2_pg_q4_b50000_r0.02_rtg_nnbaseline_HalfCheetah-v4_26-09-2022_02-52-42/*')[0]

_, q4_b50000_r0_02_returns = get_section_results(q4_b50000_r0_02)
_, q4_b50000_r0_02_rtg_returns = get_section_results(q4_b50000_r0_02_rtg)
_, q4_b50000_r0_02_nnbaseline_returns = get_section_results(q4_b50000_r0_02_nnbaseline)
_, q4_b50000_r0_02_rtg_nnbaseline_returns = get_section_results(q4_b50000_r0_02_rtg_nnbaseline)

fig7 = plt.figure()
plt.plot(itr, q4_b50000_r0_02_returns, label='vanilla')
plt.plot(itr, q4_b50000_r0_02_rtg_returns, label='rtg')
plt.plot(itr, q4_b50000_r0_02_nnbaseline_returns, label='baseline')
plt.plot(itr, q4_b50000_r0_02_rtg_nnbaseline_returns, label='rtg, baseline')

plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Half Cheetah Policy Gradient Learning Curves')
plt.legend(loc='upper left')
fig7.set_size_inches(10, 6)
plt.show()

#%% q5

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

itr5 = np.arange(300)

fig7 = plt.figure()

plt.plot(itr5, q5_b2000_r0_001_lambda1_returns, label='lambda=1')
plt.plot(itr5, q5_b2000_r0_001_lambda99_returns, label='lambda=0.99')
plt.plot(itr5, q5_b2000_r0_001_lambda98_returns, label='lambda=0.98')
plt.plot(itr5, q5_b2000_r0_001_lambda95_returns, label='lambda=0.95')
plt.plot(itr5, q5_b2000_r0_001_lambda0_returns, label='lambda=0')



plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Hopper Policy Gradient Learning Curves with Varying GAE Lambda')
plt.legend(loc='upper left')
fig7.set_size_inches(10, 6)
plt.show()

from scipy.interpolate import make_interp_spline
from scipy.ndimage.filters import gaussian_filter1d
l1_smooth = make_interp_spline(itr5, q5_b2000_r0_001_lambda1_returns)
l99_smooth = make_interp_spline(itr5, q5_b2000_r0_001_lambda99_returns)
l98_smooth = make_interp_spline(itr5, q5_b2000_r0_001_lambda98_returns)
l95_smooth = make_interp_spline(itr5, q5_b2000_r0_001_lambda95_returns)
l0_smooth = make_interp_spline(itr5, q5_b2000_r0_001_lambda0_returns)

l1 = l1_smooth(itr5)
l99 = l99_smooth(itr5)
l98 = l98_smooth(itr5)
l95 = l95_smooth(itr5)
l0 = l0_smooth(itr5)

l1 = gaussian_filter1d(q5_b2000_r0_001_lambda1_returns, sigma=2)
l99 = gaussian_filter1d(q5_b2000_r0_001_lambda99_returns, sigma=2)
l98 = gaussian_filter1d(q5_b2000_r0_001_lambda98_returns, sigma=2)
l95 = gaussian_filter1d(q5_b2000_r0_001_lambda95_returns, sigma=2)
l0 = gaussian_filter1d(q5_b2000_r0_001_lambda0_returns, sigma=2)

fig8 = plt.figure()

plt.plot(itr5, l1, label='lambda=1')
plt.plot(itr5, l99, label='lambda=0.99')
plt.plot(itr5, l98, label='lambda=0.98')
plt.plot(itr5, l95, label='lambda=0.95')
plt.plot(itr5, l0, label='lambda=0')



plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Hopper Policy Gradient Learning Curves with Varying GAE Lambda (Smoothed via Gaussian Filter)')
plt.legend(loc='upper left')
fig8.set_size_inches(10, 6)
plt.show()










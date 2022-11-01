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
    Z = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Z.append(v.simple_value)
    return X, Y, Z

def get_section_results_mean(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                Z.append(v.simple_value)
    return X, Y, Z


#%% q1
q1 = glob.glob('**/q1_MsPacman-v0_13-10-2022_22-59-43/*')[0]

steps, q1_returns, q1_means = get_section_results_mean(q1)

steps = steps[2:]
q1_returns = q1_returns[1:]

itr = np.arange(len(q1_returns))*1000

fig1 = plt.figure()
plt.plot(steps, q1_returns, label="Mean Epoch Reward")
plt.plot(steps, q1_means, label="Best Mean Reward")
plt.legend(loc = "lower right")
plt.xlabel('Iteration')
plt.ylabel('Return')
plt.title('MsPacman DQN Learning Curve')
fig1.set_size_inches(10, 6)
plt.show()

#%% q2

q2_dqn1 = glob.glob('**/q2_dqn_1_LunarLander-v3_13-10-2022_23-12-10/*')[0]
q2_dqn2 = glob.glob('**/q2_dqn_2_LunarLander-v3_13-10-2022_23-12-10/*')[0]
q2_dqn3 = glob.glob('**/q2_dqn_3_LunarLander-v3_13-10-2022_23-12-10/*')[0]

q2_ddqn1 = glob.glob('**/q2_doubledqn_1_LunarLander-v3_14-10-2022_03-06-29/*')[0]
q2_ddqn2 = glob.glob('**/q2_doubledqn_2_LunarLander-v3_14-10-2022_03-06-29/*')[0]
q2_ddqn3 = glob.glob('**/q2_doubledqn_3_LunarLander-v3_14-10-2022_03-06-29/*')[0]

steps,q2_dqn1_returns,_ = get_section_results(q2_dqn1)
_,q2_dqn2_returns,_ = get_section_results(q2_dqn2)
_,q2_dqn3_returns,_ = get_section_results(q2_dqn3)

_,q2_ddqn1_returns,_ = get_section_results(q2_ddqn1)
_,q2_ddqn2_returns,_ = get_section_results(q2_ddqn2)
_,q2_ddqn3_returns,_ = get_section_results(q2_ddqn3)

steps = np.array(steps)-1
steps = steps[1:]
itr = np.arange(len(q1_returns))*1000

q2_dqn_returns = (np.array(q2_dqn1_returns)+np.array(q2_dqn2_returns)+np.array(q2_dqn3_returns))/3
q2_ddqn_returns = (np.array(q2_ddqn1_returns)+np.array(q2_ddqn2_returns)+np.array(q2_ddqn3_returns))/3

fig1 = plt.figure()
plt.plot(steps, q2_dqn_returns, label='DQN Average Return')
plt.plot(steps, q2_ddqn_returns, label='DDQN Average Return')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Training Return')
plt.title('LunarLander DQN and DDQN Learning Curves')
fig1.set_size_inches(10, 6)
plt.show()

#%% q3

q3_hparam1 = glob.glob('**/q3_hparam1_LunarLander-v3_17-10-2022_01-45-58/*')[0]
q3_hparam2 = glob.glob('**/q3_hparam2_LunarLander-v3_17-10-2022_01-45-57/*')[0]
q3_hparam3 = glob.glob('**/q3_hparam3_LunarLander-v3_17-10-2022_01-45-58/*')[0]

steps,q3_hparam1_returns,_ = get_section_results(q3_hparam1)
_,q3_hparam2_returns,_ = get_section_results(q3_hparam2)
_,q3_hparam3_returns,_ = get_section_results(q3_hparam3)

steps = np.array(steps)-1
steps = steps[1:]
itr = np.arange(len(q1_returns))*1000

fig1 = plt.figure()
plt.plot(steps, q2_dqn1_returns, label='Batch Size: 32')
plt.plot(steps, q3_hparam1_returns, label='Batch Size: 16')
plt.plot(steps, q3_hparam2_returns, label='Batch Size: 128')
plt.plot(steps, q3_hparam3_returns, label='Batch Size: 2048')

plt.xlabel('Iteration')
plt.legend(loc='lower left')
plt.ylabel('Average Training Return')
plt.title('LunarLander DQN Learning Curves with Varying Batch Sizes')
fig1.set_size_inches(10, 6)
plt.show()

#%% q4

q4_ac_1_1 = glob.glob('**/q4_ac_1_1_CartPole-v0_14-10-2022_23-31-41/*')[0]
q4_ac_100_1 = glob.glob('**/q4_ac_1_100_CartPole-v0_14-10-2022_23-36-37/*')[0]
q4_ac_1_100 = glob.glob('**/q4_ac_100_1_CartPole-v0_14-10-2022_23-33-12/*')[0]
q4_ac_10_10 = glob.glob('**/q4_ac_10_10_CartPole-v0_14-10-2022_23-40-32/*')[0]

steps,_,q4_ac_1_1_returns = get_section_results(q4_ac_1_1)
_,_,q4_ac_100_1_returns = get_section_results(q4_ac_100_1)
_,_,q4_ac_1_100_returns = get_section_results(q4_ac_1_100)
_,_,q4_ac_10_10_returns = get_section_results(q4_ac_10_10)

steps = np.array(steps)-1

itr = np.arange(len(q1_returns))*1000

fig1 = plt.figure()
plt.plot(steps, q4_ac_1_1_returns, label='ntu 1, ngsptu 1')
plt.plot(steps, q4_ac_1_100_returns, label='ntu 1, ngsptu 100')
plt.plot(steps, q4_ac_100_1_returns, label='ntu 100, ngsptu 1')
plt.plot(steps, q4_ac_10_10_returns, label='ntu 10, ngsptu 10')

plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Evaluation Return')
plt.title('CartPole AC Learning Curves with Varying Target Updates and Gradient Steps per Target Update')
fig1.set_size_inches(10, 6)
plt.show()

#%% q5


q5_pendulum = glob.glob('**/q5_10_10_InvertedPendulum-v4_15-10-2022_00-21-46/*')[0]
q5_cheetah = glob.glob('**/q5_10_10_HalfCheetah-v4_15-10-2022_00-21-46/*')[0]


steps_pen,_,q5_pendulum_returns = get_section_results(q5_pendulum)
steps_cheetah,_,q5_cheetah_returns = get_section_results(q5_cheetah)


steps_pen = np.array(steps_pen)
steps_cheetah = np.array(steps_cheetah)

itr = np.arange(len(q1_returns))*1000

fig1 = plt.figure()
plt.plot(steps_pen, q5_pendulum_returns)


plt.xlabel('Iteration')
plt.ylabel('Average Evaluation Return')
plt.title('InvertedPendulum AC Learning Curve')
fig1.set_size_inches(10, 6)
plt.show()

fig1 = plt.figure()
plt.plot(steps_cheetah, q5_cheetah_returns)


plt.xlabel('Iteration')
plt.ylabel('Average Evaluation Return')
plt.title('HalfCheetah AC Learning Curve')
fig1.set_size_inches(10, 6)
plt.show()

#%% q6

q6_pendulum = glob.glob('**/q6a_sac_InvertedPendulum_InvertedPendulum-v4_19-10-2022_16-43-25/*')[0]
q6_cheetah = glob.glob('**/q6b_sac_HalfCheetah_HalfCheetah-v4_19-10-2022_04-49-45/*')[0]


steps_pen,_,q6_pendulum_returns = get_section_results(q6_pendulum)
steps_cheetah,_,q6_cheetah_returns = get_section_results(q6_cheetah)


steps_pen = np.array(steps_pen)
steps_cheetah = np.array(steps_cheetah)

itr = np.arange(len(q1_returns))*1000

fig1 = plt.figure()
plt.plot(steps_pen, q6_pendulum_returns)


plt.xlabel('Iteration')
plt.ylabel('Average Evaluation Return')
plt.title('InvertedPendulum SAC Learning Curve')
fig1.set_size_inches(10, 6)
plt.show()

fig1 = plt.figure()
plt.plot(steps_cheetah, q6_cheetah_returns)


plt.xlabel('Iteration')
plt.ylabel('Average Evaluation Return')
plt.title('HalfCheetah SAC Learning Curve')
fig1.set_size_inches(10, 6)
plt.show()




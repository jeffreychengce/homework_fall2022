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


#%% q2

q2 = glob.glob('**/hw4_q2_obstacles_singleiteration_obstacles-cs285-v0_30-10-2022_19-53-25/*')[0]




steps,q2_train,q2_eval = get_section_results(q2)


itr = np.arange(len(q2_eval))

fig1 = plt.figure()
plt.scatter(itr, q2_train, label='Training Return')
plt.scatter(itr, q2_eval, label='Evaluation Return')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Return')
plt.title('MPC in Obstacles Environment')
fig1.set_size_inches(10, 6)
plt.show()

#%% q3

q3_obstacles = glob.glob('**/hw4_q3_obstacles_obstacles-cs285-v0_30-10-2022_19-53-44/*')[0]
q3_reacher = glob.glob('**/hw4_q3_reacher_reacher-cs285-v0_30-10-2022_20-06-57/*')[0]
q3_cheetah = glob.glob('**/hw4_q3_cheetah_cheetah-cs285-v0_30-10-2022_21-06-51/*')[0]

steps,q3_obstacles_train,q3_obstacles_eval = get_section_results(q3_obstacles)
_,q3_reacher_train,q3_reacher_eval = get_section_results(q3_reacher)
_,q3_cheetah_train,q3_cheetah_eval = get_section_results(q3_cheetah)

itr_obstacles = np.arange(len(q3_obstacles_eval))
itr_reacher = np.arange(len(q3_reacher_eval))
itr_cheetah = np.arange(len(q3_cheetah_eval))

fig1 = plt.figure()
#plt.plot(itr_obstacles, q3_obstacles_train, label='Training Returns')
plt.plot(itr_obstacles, q3_obstacles_eval, label='Evaluation Returns')
plt.xlabel('Iteration')
#plt.legend(loc='lower left')
plt.ylabel('Average Evaluation Return')
plt.title('On-Policy MBRL in the Obstacles Environment')
fig1.set_size_inches(10, 6)
plt.show()

fig1 = plt.figure()
#plt.plot(itr_reacher, q3_reacher_train, label='Training Returns')
plt.plot(itr_reacher, q3_reacher_eval, label='Evaluation Returns')
plt.xlabel('Iteration')
#plt.legend(loc='lower left')
plt.ylabel('Average Evaluation Return')
plt.title('On-Policy MBRL in the Reacher Environment')
fig1.set_size_inches(10, 6)
plt.show()

fig1 = plt.figure()
#plt.plot(itr_cheetah, q3_cheetah_train, label='Training Returns')
plt.plot(itr_cheetah, q3_cheetah_eval, label='Evaluation Returns')
plt.xlabel('Iteration')
#plt.legend(loc='lower left')
plt.ylabel('Average Evaluation Return')
plt.title('On-Policy MBRL in the Cheetah Environment')
fig1.set_size_inches(10, 6)
plt.show()

#%% q4

q4_horizon5 = glob.glob('**/hw4_q4_reacher_horizon5_reacher-cs285-v0_31-10-2022_00-48-42/*')[0]
q4_horizon15 = glob.glob('**/hw4_q4_reacher_horizon15_reacher-cs285-v0_31-10-2022_00-48-42/*')[0]
q4_horizon30 = glob.glob('**/hw4_q4_reacher_horizon30_reacher-cs285-v0_31-10-2022_00-48-42/*')[0]

q4_numseq100 = glob.glob('**/hw4_q4_reacher_numseq100_reacher-cs285-v0_31-10-2022_12-24-54/*')[0]
q4_numseq1000 = glob.glob('**/hw4_q4_reacher_numseq1000_reacher-cs285-v0_31-10-2022_12-24-54/*')[0]

q4_ensemble1 = glob.glob('**/hw4_q4_reacher_ensemble1_reacher-cs285-v0_31-10-2022_09-11-49/*')[0]
q4_ensemble3 = glob.glob('**/hw4_q4_reacher_ensemble3_reacher-cs285-v0_31-10-2022_09-11-49/*')[0]
q4_ensemble5 = glob.glob('**/hw4_q4_reacher_ensemble5_reacher-cs285-v0_31-10-2022_09-11-49/*')[0]

_,_,q4_horizon5_eval = get_section_results(q4_horizon5)
_,_,q4_horizon15_eval = get_section_results(q4_horizon15)
_,_,q4_horizon30_eval = get_section_results(q4_horizon30)

_,_,q4_numseq100_eval = get_section_results(q4_numseq100)
_,_,q4_numseq1000_eval = get_section_results(q4_numseq1000)

_,_,q4_ensemble1_eval = get_section_results(q4_ensemble1)
_,_,q4_ensemble3_eval = get_section_results(q4_ensemble3)
_,_,q4_ensemble5_eval = get_section_results(q4_ensemble5)

itr = np.arange(15)

fig1 = plt.figure()
plt.plot(itr, q4_horizon5_eval, label='Horizon: 5')
plt.plot(itr, q4_horizon15_eval, label='Horizon: 15')
plt.plot(itr, q4_horizon30_eval, label='Horizon: 30')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Evaluation Return')
plt.title('On-Policy MBRL in the Reacher Environment, Varying Planning Horizons')
fig1.set_size_inches(10, 6)
plt.show()

fig1 = plt.figure()
plt.plot(itr, q4_numseq100_eval, label='Candidate Sequences: 100')
plt.plot(itr, q4_numseq1000_eval, label='Candidate Sequences: 1000')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Evaluation Return')
plt.title('On-Policy MBRL in the Reacher Environment, Varying Candidate Action Sequences')
fig1.set_size_inches(10, 6)
plt.show()


fig1 = plt.figure()
plt.plot(itr, q4_ensemble1_eval, label='Ensemble Size: 5')
plt.plot(itr, q4_ensemble3_eval, label='Ensemble Size: 15')
plt.plot(itr, q4_ensemble5_eval, label='Ensemble Size: 30')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Evaluation Return')
plt.title('On-Policy MBRL in the Reacher Environment, Varying Ensemble Sizes')
fig1.set_size_inches(10, 6)
plt.show()
#%% q5

q5_random = glob.glob('**/hw4_q5_cheetah_random_cheetah-cs285-v0_31-10-2022_17-07-35/*')[0]
q5_cem2 = glob.glob('**/hw4_q5_cheetah_cem_2_cheetah-cs285-v0_31-10-2022_17-07-35/*')[0]
q5_cem4 = glob.glob('**/hw4_q5_cheetah_cem_4_cheetah-cs285-v0_31-10-2022_11-19-10/*')[0]


_,_,q5_random_eval = get_section_results(q5_random)
_,_,q5_cem2_eval = get_section_results(q5_cem2)
_,_,q5_cem4_eval = get_section_results(q5_cem4)



itr = np.arange(5)

fig1 = plt.figure()
plt.plot(itr, q5_random_eval, label='Random Shooting')
plt.plot(itr, q5_cem2_eval, label='2 CEM Iterations')
plt.plot(itr, q5_cem4_eval, label='4 CEM Iterations')
plt.xlabel('Iteration')
plt.legend(loc='upper left')
plt.ylabel('Average Evaluation Return')
plt.title('On-Policy MBRL in the Cheetah Environment, Varying Action Selection')
fig1.set_size_inches(10, 6)
plt.show()



#%% q6

q6_SAC = glob.glob('**/hw4_q5_cheetah_random_cheetah-cs285-v0_31-10-2022_17-07-35/*')[0]
q6_Dyna = glob.glob('**/hw4_q5_cheetah_cem_2_cheetah-cs285-v0_31-10-2022_17-07-35/*')[0]
q6_MBPO = glob.glob('**/hw4_q5_cheetah_cem_4_cheetah-cs285-v0_31-10-2022_11-19-10/*')[0]


_,_,q6_SAC_eval = get_section_results(q6_SAC)
_,_,q6_Dyna_eval = get_section_results(q6_Dyna)
_,_,q6_MBPO_eval = get_section_results(q6_MBPO)



itr = np.arange(5)

fig1 = plt.figure()
plt.plot(itr, q6_SAC_eval, label='Model-Free SAC')
plt.plot(itr, q6_Dyna_eval, label='Dyna: Single Step Rollouts')
plt.plot(itr, q6_MBPO_eval, label='MBPO: 10 Step Rollouts')
plt.xlabel('Iteration')
plt.legend(loc='upper left')
plt.ylabel('Average Evaluation Return')
plt.title('Variations of MBPO in the Cheetah Environment')
fig1.set_size_inches(10, 6)
plt.show()




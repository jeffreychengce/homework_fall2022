# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:28:22 2022

@author: Jeffrey
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import glob
tf.data.experimental.ignore_errors()
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
q1_easy_rnd = glob.glob('**/hw5_expl_q1_env1_rnd_PointmassEasy-v0_21-11-2022_16-35-21/*')[5]
q1_easy_rand = glob.glob('**/hw5_expl_q1_env1_random_PointmassEasy-v0_21-11-2022_16-35-21/*')[5]


steps,train,q1_easy_rnd = get_section_results(q1_easy_rnd)
steps,_,q1_easy_rand = get_section_results(q1_easy_rand)

fig1 = plt.figure()
plt.plot(steps, q1_easy_rnd, label='RND')
plt.plot(steps, q1_easy_rand, label='Random')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Return')
plt.title('Learning Curves for PointmassEasy')
fig1.set_size_inches(10, 6)
plt.show()

q1_med_rnd = glob.glob('**/hw5_expl_q1_env2_rnd_PointmassMedium-v0_21-11-2022_16-20-20/*')[5]
q1_med_rand = glob.glob('**/hw5_expl_q1_env2_random_PointmassMedium-v0_21-11-2022_16-35-21/*')[5]


steps,train,q1_med_rnd = get_section_results(q1_med_rnd)
steps,_,q1_med_rand = get_section_results(q1_med_rand)

fig1 = plt.figure()
plt.plot(steps, q1_med_rnd, label='RND')
plt.plot(steps, q1_med_rand, label='Random')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Return')
plt.title('Learning Curves for PointmassMedium')
fig1.set_size_inches(10, 6)
plt.show()

q1_med_rnd = glob.glob('**/hw5_expl_q1_env2_rnd_PointmassMedium-v0_21-11-2022_16-20-20/*')[5]
q1_med_boltz = glob.glob('**/hw5_expl_q1_alg_med_PointmassMedium-v0_21-11-2022_17-24-25/*')[5]


steps,train,q1_med_rnd = get_section_results(q1_med_rnd)
steps,_,q1_med_boltz = get_section_results(q1_med_boltz)

fig1 = plt.figure()
plt.plot(steps, q1_med_rnd, label='RND')
plt.plot(steps, q1_med_boltz, label='Boltzmann')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Return')
plt.title('Learning Curves for PointmassMedium')
fig1.set_size_inches(10, 6)
plt.show()

q1_hard_rnd = glob.glob('**/hw5_expl_q1_hard_rnd_PointmassHard-v0_21-11-2022_17-24-25/*')[5]
q1_hard_boltz = glob.glob('**/hw5_expl_q1_alg_hard_PointmassHard-v0_21-11-2022_17-24-25/*')[5]


steps,train,q1_hard_rnd = get_section_results(q1_hard_rnd)
steps,_,q1_hard_boltz = get_section_results(q1_hard_boltz)

fig1 = plt.figure()
plt.plot(steps, q1_hard_rnd, label='RND')
plt.plot(steps, q1_hard_boltz, label='Boltzmann')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Return')
plt.title('Learning Curves for PointmassHard')
fig1.set_size_inches(10, 6)
plt.show()

#%% q2
q2_med_dql = glob.glob('**/hw5_expl_q2_dqn_PointmassMedium-v0_21-11-2022_03-14-13/*')[5]
q2_med_cql = glob.glob('**/hw5_expl_q2_cql_PointmassMedium-v0_21-11-2022_03-14-13/*')[5]

steps,train,q2_med_dql = get_section_results(q2_med_dql)
steps,_,q2_med_cql = get_section_results(q2_med_cql)

fig1 = plt.figure()
plt.plot(steps, q2_med_dql, label='DQN')
plt.plot(steps, q2_med_cql, label='CQL')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Return')
plt.title('Learning Curves for PointmassMedium')
fig1.set_size_inches(10, 6)
plt.show()

q2_5000_dql = glob.glob('**/hw5_expl_q2_dqn_PointmassMedium-v0_21-11-2022_17-49-31/*')[5]
q2_15000_dql = glob.glob('**/hw5_expl_q2_cql_PointmassMedium-v0_21-11-2022_17-49-31/*')[5]

steps,train,q2_5000_dql = get_section_results(q2_5000_dql)
steps,_,q2_15000_dql = get_section_results(q2_15000_dql)

fig1 = plt.figure()
plt.plot(steps, q2_5000_dql, label='5,000')
plt.plot(steps, q2_med_dql, label='10,000')
plt.plot(steps, q2_15000_dql, label='15,000')

plt.xlabel('Iteration')
plt.legend(loc='lower right', title='Number of Exploration Steps')
plt.ylabel('Average Return')
plt.title('Learning Curves for DQN in PointmassMedium, Varying Exploration Steps')
fig1.set_size_inches(10, 6)
plt.show()

q2_5000_cql = glob.glob('**/hw5_expl_q2_dqn_numsteps_5000_PointmassMedium-v0_21-11-2022_18-08-44/*')[5]
q2_15000_cql = glob.glob('**/hw5_expl_q2_dqn_numsteps_15000_PointmassMedium-v0_21-11-2022_18-08-44/*')[5]

steps,train,q2_5000_cql = get_section_results(q2_5000_cql)
steps,_,q2_15000_cql = get_section_results(q2_15000_cql)

fig1 = plt.figure()
plt.plot(steps, q2_5000_cql, label='5,000')
plt.plot(steps, q2_med_cql, label='10,000')
plt.plot(steps, q2_15000_cql, label='15,000')

plt.xlabel('Iteration')
plt.legend(loc='lower right', title='Number of Exploration Steps')
plt.ylabel('Average Return')
plt.title('Learning Curves for CQL in PointmassMedium, Varying Exploration Steps')
fig1.set_size_inches(10, 6)
plt.show()

q2_alpha002 = glob.glob('**/hw5_expl_q2_alpha0.02_PointmassMedium-v0_21-11-2022_18-36-49/*')[5]
q2_alpha05 = glob.glob('**/hw5_expl_q2_alpha0.5_PointmassMedium-v0_21-11-2022_18-36-49/*')[5]

steps,train,q2_alpha002 = get_section_results(q2_alpha002)
steps,_,q2_alpha05 = get_section_results(q2_alpha05)

fig1 = plt.figure()
plt.plot(steps, q2_alpha002, label='0.02')
plt.plot(steps, q2_med_cql, label='0.1')
plt.plot(steps, q2_alpha05, label='0.5')

plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Return')
plt.title('Learning Curves for PointmassMedium, Varying CQL Alphas')
fig1.set_size_inches(10, 6)
plt.show()


#%% q3
q3_med_dql = glob.glob('**/hw5_expl_q3_medium_dqn_PointmassMedium-v0_21-11-2022_18-36-49/*')[5]
q3_med_cql = glob.glob('**/hw5_expl_q3_medium_cql_PointmassMedium-v0_21-11-2022_18-36-49/*')[5]

steps,train,q3_med_dql = get_section_results(q3_med_dql)
steps,_,q3_med_cql = get_section_results(q3_med_cql)

fig1 = plt.figure()
plt.plot(steps, q3_med_dql, label='DQN')
plt.plot(steps, q3_med_cql, label='CQL')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Return')
plt.title('Learning Curves for PointmassMedium')
fig1.set_size_inches(10, 6)
plt.show()

q3_hard_dql = glob.glob('**/hw5_expl_q3_hard_dqn_PointmassHard-v0_21-11-2022_18-36-49/*')[5]
q3_hard_cql = glob.glob('**/hw5_expl_q3_hard_cql_PointmassHard-v0_21-11-2022_18-36-49/*')[5]

steps,train,q3_hard_dql = get_section_results(q3_hard_dql)
steps,_,q3_hard_cql = get_section_results(q3_hard_cql)

fig1 = plt.figure()
plt.plot(steps, q3_hard_dql, label='DQN')
plt.plot(steps, q3_hard_cql, label='CQL')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Return')
plt.title('Learning Curves for PointmassHard')
fig1.set_size_inches(10, 6)
plt.show()

#%% q4

q4_es_01 = glob.glob('**/hw5_expl_q4_awac_easy_supervised_lam0.1_PointmassEasy-v0_21-11-2022_22-23-51/*')[5]
q4_es_1 = glob.glob('**/hw5_expl_q4_awac_easy_supervised_lam1_PointmassEasy-v0_21-11-2022_22-23-51/*')[5]
q4_es_2 = glob.glob('**/hw5_expl_q4_awac_easy_supervised_lam2_PointmassEasy-v0_21-11-2022_22-23-51/*')[5]
q4_es_10 = glob.glob('**/hw5_expl_q4_awac_easy_supervised_lam10_PointmassEasy-v0_21-11-2022_22-23-51/*')[5]
q4_es_20 = glob.glob('**/hw5_expl_q4_awac_easy_supervised_lam20_PointmassEasy-v0_21-11-2022_22-23-52/*')[5]
q4_es_50 = glob.glob('**/hw5_expl_q4_awac_easy_supervised_lam50_PointmassEasy-v0_21-11-2022_22-23-52/*')[5]

steps,_,q4_es_01 = get_section_results(q4_es_01)
steps,_,q4_es_1 = get_section_results(q4_es_1)
steps,_,q4_es_2 = get_section_results(q4_es_2)
steps,_,q4_es_10 = get_section_results(q4_es_10)
steps,_,q4_es_20 = get_section_results(q4_es_20)
steps,_,q4_es_50 = get_section_results(q4_es_50)

fig1 = plt.figure()
plt.plot(steps, q4_es_01, label='0.1')
plt.plot(steps, q4_es_1, label='1')
plt.plot(steps, q4_es_2, label='2')
plt.plot(steps, q4_es_10, label='10')
plt.plot(steps, q4_es_20, label='20')
plt.plot(steps, q4_es_50, label='50')
plt.xlabel('Iteration')
plt.legend(loc='lower right', title='Lambda')
plt.ylabel('Average Return')
plt.title('AWAC (Supervised) Learning Curves for PointmassEasy, Varying Lambdas')
fig1.set_size_inches(10, 6)
plt.show()

q4_eu_01 = glob.glob('**/hw5_expl_q4_awac_easy_unsupervised_lam0.1_PointmassEasy-v0_21-11-2022_22-23-50/*')[5]
q4_eu_1 = glob.glob('**/hw5_expl_q4_awac_easy_unsupervised_lam1_PointmassEasy-v0_21-11-2022_22-23-50/*')[5]
q4_eu_2 = glob.glob('**/hw5_expl_q4_awac_easy_unsupervised_lam2_PointmassEasy-v0_21-11-2022_22-23-51/*')[5]
q4_eu_10 = glob.glob('**/hw5_expl_q4_awac_easy_unsupervised_lam10_PointmassEasy-v0_21-11-2022_22-23-51/*')[5]
q4_eu_20 = glob.glob('**/hw5_expl_q4_awac_easy_unsupervised_lam20_PointmassEasy-v0_21-11-2022_22-23-51/*')[5]
q4_eu_50 = glob.glob('**/hw5_expl_q4_awac_easy_unsupervised_lam50_PointmassEasy-v0_21-11-2022_22-23-51/*')[5]


steps,_,q4_eu_01 = get_section_results(q4_eu_01)
steps,_,q4_eu_1 = get_section_results(q4_eu_1)
steps,_,q4_eu_2 = get_section_results(q4_eu_2)
steps,_,q4_eu_10 = get_section_results(q4_eu_10)
steps,_,q4_eu_20 = get_section_results(q4_eu_20)
steps,_,q4_eu_50 = get_section_results(q4_eu_50)

fig1 = plt.figure()
plt.plot(steps, q4_eu_01, label='0.1')
plt.plot(steps, q4_eu_1, label='1')
plt.plot(steps, q4_eu_2, label='2')
plt.plot(steps, q4_eu_10, label='10')
plt.plot(steps, q4_eu_20, label='20')
plt.plot(steps, q4_eu_50, label='50')
plt.xlabel('Iteration')
plt.legend(loc='lower right', title='Lambda')
plt.ylabel('Average Return')
plt.title('AWAC (Unsupervised) Learning Curves for PointmassEasy, Varying Lambdas')
fig1.set_size_inches(10, 6)
plt.show()

q4_ms_01 = glob.glob('**/hw5_expl_q4_awac_medium_supervised_lam0.1_PointmassMedium-v0_21-11-2022_22-23-51/*')[5]
q4_ms_1 = glob.glob('**/hw5_expl_q4_awac_medium_supervised_lam1_PointmassMedium-v0_21-11-2022_22-23-51/*')[5]
q4_ms_2 = glob.glob('**/hw5_expl_q4_awac_medium_supervised_lam2_PointmassMedium-v0_21-11-2022_22-23-51/*')[5]
q4_ms_10 = glob.glob('**/hw5_expl_q4_awac_medium_supervised_lam10_PointmassMedium-v0_21-11-2022_22-23-50/*')[5]
q4_ms_20 = glob.glob('**/hw5_expl_q4_awac_medium_supervised_lam20_PointmassMedium-v0_21-11-2022_22-23-53/*')[5]
q4_ms_50 = glob.glob('**/hw5_expl_q4_awac_medium_supervised_lam50_PointmassMedium-v0_21-11-2022_22-23-51/*')[5]

steps,_,q4_ms_01 = get_section_results(q4_ms_01)
steps,_,q4_ms_1 = get_section_results(q4_ms_1)
steps,_,q4_ms_2 = get_section_results(q4_ms_2)
steps,_,q4_ms_10 = get_section_results(q4_ms_10)
steps,_,q4_ms_20 = get_section_results(q4_ms_20)
steps,_,q4_ms_50 = get_section_results(q4_ms_50)

fig1 = plt.figure()
plt.plot(steps, q4_ms_01, label='0.1')
plt.plot(steps, q4_ms_1, label='1')
plt.plot(steps, q4_ms_2, label='2')
plt.plot(steps, q4_ms_10, label='10')
plt.plot(steps, q4_ms_20, label='20')
plt.plot(steps, q4_ms_50, label='50')
plt.xlabel('Iteration')
plt.legend(loc='lower right', title='Lambda')
plt.ylabel('Average Return')
plt.title('AWAC (Supervised) Learning Curves for PointmassMedium, Varying Lambdas')
fig1.set_size_inches(10, 6)
plt.show()

q4_mu_01 = glob.glob('**/hw5_expl_q4_awac_medium_unsupervised_lam0.1_PointmassMedium-v0_21-11-2022_22-23-51/*')[5]
q4_mu_1 = glob.glob('**/hw5_expl_q4_awac_medium_unsupervised_lam1_PointmassMedium-v0_21-11-2022_22-23-51/*')[5]
q4_mu_2 = glob.glob('**/hw5_expl_q4_awac_medium_unsupervised_lam2_PointmassMedium-v0_21-11-2022_22-23-50/*')[5]
q4_mu_10 = glob.glob('**/hw5_expl_q4_awac_medium_unsupervised_lam10_PointmassMedium-v0_21-11-2022_22-23-51/*')[5]
q4_mu_20 = glob.glob('**/hw5_expl_q4_awac_medium_unsupervised_lam20_PointmassMedium-v0_21-11-2022_22-23-51/*')[5]
q4_mu_50 = glob.glob('**/hw5_expl_q4_awac_medium_unsupervised_lam50_PointmassMedium-v0_21-11-2022_22-23-51/*')[5]


steps,_,q4_mu_01 = get_section_results(q4_mu_01)
steps,_,q4_mu_1 = get_section_results(q4_mu_1)
steps,_,q4_mu_2 = get_section_results(q4_mu_2)
steps,_,q4_mu_10 = get_section_results(q4_mu_10)
steps,_,q4_mu_20 = get_section_results(q4_mu_20)
steps,_,q4_mu_50 = get_section_results(q4_mu_50)

fig1 = plt.figure()
plt.plot(steps, q4_mu_01, label='0.1')
plt.plot(steps, q4_mu_1, label='1')
plt.plot(steps, q4_mu_2, label='2')
plt.plot(steps, q4_mu_10, label='10')
plt.plot(steps, q4_mu_20, label='20')
plt.plot(steps, q4_mu_50, label='50')
plt.xlabel('Iteration')
plt.legend(loc='lower right', title='Lambda')
plt.ylabel('Average Return')
plt.title('AWAC (Unsupervised) Learning Curves for PointmassMedium, Varying Lambdas')
fig1.set_size_inches(10, 6)
plt.show()

#%% q5
q5_es_5 = glob.glob('**/hw5_expl_q5_iql_easy_supervised_lam1_tau0.5_PointmassEasy-v0_22-11-2022_19-31-02/*')[5]
q5_es_7 = glob.glob('**/hw5_expl_q5_iql_easy_supervised_lam1_tau0.7_PointmassEasy-v0_22-11-2022_19-31-02/*')[5]
q5_es_9 = glob.glob('**/hw5_expl_q5_iql_easy_supervised_lam1_tau0.9_PointmassEasy-v0_22-11-2022_19-31-02/*')[5]
q5_es_99 = glob.glob('**/hw5_expl_q5_iql_easy_supervised_lam1_tau0.99_PointmassEasy-v0_22-11-2022_19-31-02/*')[5]

steps,_,q5_es_5 = get_section_results(q5_es_5)
steps,_,q5_es_7 = get_section_results(q5_es_7)
steps,_,q5_es_9 = get_section_results(q5_es_9)
steps,_,q5_es_99 = get_section_results(q5_es_99)


fig1 = plt.figure()
plt.plot(steps, q5_es_5, label='0.5')
plt.plot(steps, q5_es_7, label='0.7')
plt.plot(steps, q5_es_9, label='0.9')
plt.plot(steps, q5_es_99, label='0.99')
plt.xlabel('Iteration')
plt.legend(loc='lower right', title='Tau')
plt.ylabel('Average Return')
plt.title('IQL (Supervised) Learning Curves for PointmassEasy, Varying Tau Values, Lambda=1')
fig1.set_size_inches(10, 6)
plt.show()

q5_eu_5 = glob.glob('**/hw5_expl_q5_iql_easy_unsupervised_lam2_tau0.5_PointmassEasy-v0_22-11-2022_19-31-02/*')[5]
q5_eu_7 = glob.glob('**/hw5_expl_q5_iql_easy_unsupervised_lam2_tau0.7_PointmassEasy-v0_22-11-2022_19-31-02/*')[5]
q5_eu_9 = glob.glob('**/hw5_expl_q5_iql_easy_unsupervised_lam2_tau0.9_PointmassEasy-v0_22-11-2022_19-31-02/*')[5]
q5_eu_99 = glob.glob('**/hw5_expl_q5_iql_easy_unsupervised_lam2_tau0.99_PointmassEasy-v0_22-11-2022_19-31-02/*')[5]

steps,_,q5_eu_5 = get_section_results(q5_eu_5)
steps,_,q5_eu_7 = get_section_results(q5_eu_7)
steps,_,q5_eu_9 = get_section_results(q5_eu_9)
steps,_,q5_eu_99 = get_section_results(q5_eu_99)


fig1 = plt.figure()
plt.plot(steps, q5_eu_5, label='0.5')
plt.plot(steps, q5_eu_7, label='0.7')
plt.plot(steps, q5_eu_9, label='0.9')
plt.plot(steps, q5_eu_99, label='0.99')
plt.xlabel('Iteration')
plt.legend(loc='lower right', title='Tau')
plt.ylabel('Average Return')
plt.title('IQL (Unsupervised) Learning Curves for PointmassEasy, Varying Tau Values, Lambda=2')
fig1.set_size_inches(10, 6)
plt.show()

q5_ms_5 = glob.glob('**/hw5_expl_q5_iql_medium_supervised_lam20_tau0.5_PointmassMedium-v0_22-11-2022_19-31-02/*')[5]
q5_ms_7 = glob.glob('**/hw5_expl_q5_iql_medium_supervised_lam20_tau0.7_PointmassMedium-v0_22-11-2022_19-31-02/*')[5]
q5_ms_9 = glob.glob('**/hw5_expl_q5_iql_medium_supervised_lam20_tau0.9_PointmassMedium-v0_22-11-2022_19-31-02/*')[5]
q5_ms_99 = glob.glob('**/hw5_expl_q5_iql_medium_supervised_lam20_tau0.99_PointmassMedium-v0_22-11-2022_19-31-02/*')[5]

steps,_,q5_ms_5 = get_section_results(q5_ms_5)
steps,_,q5_ms_7 = get_section_results(q5_ms_7)
steps,_,q5_ms_9 = get_section_results(q5_ms_9)
steps,_,q5_ms_99 = get_section_results(q5_ms_99)


fig1 = plt.figure()
plt.plot(steps, q5_ms_5, label='0.5')
plt.plot(steps, q5_ms_7, label='0.7')
plt.plot(steps, q5_ms_9, label='0.9')
plt.plot(steps, q5_ms_99, label='0.99')
plt.xlabel('Iteration')
plt.legend(loc='lower right', title='Tau')
plt.ylabel('Average Return')
plt.title('IQL (Supervised) Learning Curves for PointmassMedium, Varying Tau Values, Lambda=20')
fig1.set_size_inches(10, 6)
plt.show()

q5_mu_5 = glob.glob('**/hw5_expl_q5_iql_medium_unsupervised_lam1_tau0.5_PointmassMedium-v0_22-11-2022_19-31-02/*')[5]
q5_mu_7 = glob.glob('**/hw5_expl_q5_iql_medium_unsupervised_lam1_tau0.7_PointmassMedium-v0_22-11-2022_19-31-02/*')[5]
q5_mu_9 = glob.glob('**/hw5_expl_q5_iql_medium_unsupervised_lam1_tau0.9_PointmassMedium-v0_22-11-2022_19-31-02/*')[5]
q5_mu_99 = glob.glob('**/hw5_expl_q5_iql_medium_unsupervised_lam1_tau0.99_PointmassMedium-v0_22-11-2022_19-31-02/*')[5]

steps,_,q5_mu_5 = get_section_results(q5_mu_5)
steps,_,q5_mu_7 = get_section_results(q5_mu_7)
steps,_,q5_mu_9 = get_section_results(q5_mu_9)
steps,_,q5_mu_99 = get_section_results(q5_mu_99)


fig1 = plt.figure()
plt.plot(steps, q5_mu_5, label='0.5')
plt.plot(steps, q5_mu_7, label='0.7')
plt.plot(steps, q5_mu_9, label='0.9')
plt.plot(steps, q5_mu_99, label='0.99')
plt.xlabel('Iteration')
plt.legend(loc='lower right', title='Tau')
plt.ylabel('Average Return')
plt.title('IQL (Unsupervised) Learning Curves for PointmassMedium, Varying Tau Values, Lambda=1')
fig1.set_size_inches(10, 6)
plt.show()

fig1 = plt.figure()
plt.plot(steps, q2_med_cql, label='CQL')
plt.plot(steps, q4_mu_1, label='AWAC')
plt.plot(steps, q5_mu_99, label='IQL')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.ylabel('Average Return')
plt.title('Offline Learning Curves for PointmassMedium')
fig1.set_size_inches(10, 6)
plt.show()
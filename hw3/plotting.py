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
q1 = glob.glob('**/q1_LunarLander-v3_12-10-2022_23-17-09/*')[0]

_, q1_returns = get_section_results(q1)



itr = len(q1_returns)

fig1 = plt.figure()
plt.plot(itr, q1_returns)
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('CartPole Small Batch Policy Gradient Learning Curves')
plt.legend(loc='lower right')
fig1.set_size_inches(10, 6)
plt.show()










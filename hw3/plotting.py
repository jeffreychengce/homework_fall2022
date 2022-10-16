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
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
    return X, Y


#%% q1
q1 = glob.glob('**/q1_MsPacman-v0_13-10-2022_22-59-43/*')[0]

_, q1_returns = get_section_results(q1)



itr = np.arange(len(q1_returns))

fig1 = plt.figure()
plt.plot(itr, q1_returns)
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('MsPacman DQN Learning Curve')
plt.legend(loc='lower right')
fig1.set_size_inches(10, 6)
plt.show()










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
q1_sb_no_rtg_dsa_CartPole = glob.glob('**/q2_pg_q1_sb_no_rtg_dsa_CartPole-v0_24-09-2022_00-35-05/*')[0]
q1_sb_rtg_dsa_CartPole = glob.glob('**/q2_pg_q1_sb_rtg_dsa_CartPole-v0_24-09-2022_00-32-54/*')[0]
_, q1_sb_no_rtg_dsa_CartPole_returns = get_section_results( q1_sb_no_rtg_dsa_CartPole)
_, q1_sb_rtg_dsa_CartPole_returns = get_section_results(q1_sb_rtg_dsa_CartPole)
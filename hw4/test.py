# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:09:46 2022

@author: Jeffrey
"""

import numpy as np

n = 10

indices = np.arange(n)
rewards = np.random.random(n)*10
# indices = np.expand_dims(indices, axis=1)
# rewards = np.expand_dims(rewards, axis=1

elite_indices = np.argsort(rewards)[-4:]
elites = rewards[elite_indices]

matrix = np.random.rand(10,4,2)
elites2 = matrix[elite_indices]

means = np.mean(matrix, axis=0)
std = np.std(matrix, axis=0)

actions = np.random.normal(means, std, size = (1,4,2))

action = actions[0,0,:] == actions[0,0,:]

model_obs = []
for i in range(5):
    next_ob = np.random.rand(4)
    model_obs.append(next_ob)

next_ob = np.mean(model_obs,axis=0)

variance = np.random.rand(4,2)

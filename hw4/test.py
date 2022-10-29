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

elite_indices = np.argsort(rewards, axis=0)[-4:]
elites = rewards[elite_indices]

matrix = np.random.rand(10,4,2)

means = np.mean(matrix, axis=0)


# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:28:22 2022

@author: Jeffrey
"""

import numpy as np
import matplotlib.pyplot as plt
#%%

ant_dagger_train_avg = 4713.6533203125
ant_dagger_eval_avg = [4085.021240234375,4411.96728515625,4662.5791015625,\
                       4659.9609375,4627.04638671875,4751.5400390625,\
                       4684.8857421875,4750.33984375,4795.4150390625,\
                       4653.07470703125]
ant_dagger_eval_std = [479.2840270996094,53.54607391357422,97.1742172241211,\
                       75.70559692382812,56.39665985107422,148.15841674804688,\
                       144.58396911621094,96.27149200439453,92.31314086914062,\
                       118.64019012451172]

ant_dagger_eval_avg_norm = np.array(ant_dagger_eval_avg)/ant_dagger_train_avg
ant_dagger_eval_std_norm = np.array(ant_dagger_eval_std)/ant_dagger_train_avg

fig = plt.figure()
x = np.arange(10)+1



plt.hlines(ant_dagger_eval_avg[0],0,10,'b',label='Behavioral Cloning')
plt.hlines(ant_dagger_train_avg,0,10,'r',label='Expert')
plt.errorbar(x, ant_dagger_eval_avg, yerr=ant_dagger_eval_std, label='DAgger')
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Ant Environment Imitation Learning')

plt.legend(loc='lower right')

fig.set_size_inches(10, 6)

#%%
    
hopper_dagger_train_avg = 3772.67041015625
hopper_dagger_eval_avg = [917.3411254882812,902.77587890625,2677.923583984375,\
                          3007.51904296875,3048.494873046875,2328.36328125,\
                          2566.5517578125,3697.83447265625,3702.7109375,\
                          3708.642578125]
hopper_dagger_eval_std = [186.062255859375,80.46642303466797,901.8541870117188,\
                          1060.8375244140625,789.5468139648438,1076.2208251953125,\
                          803.6219482421875,8.018284797668457,10.15399169921875,\
                          6.562127590179443]



hopper_dagger_eval_avg_norm = np.array(hopper_dagger_eval_avg)/hopper_dagger_train_avg
hopper_dagger_eval_std_norm = np.array(hopper_dagger_eval_std)/hopper_dagger_train_avg
    
    
fig = plt.figure()
x = np.arange(10)+1



plt.hlines(hopper_dagger_eval_avg[0],0,10,'b',label='Behavioral Cloning')
plt.hlines(hopper_dagger_train_avg,0,10,'r',label='Expert')
plt.errorbar(x, hopper_dagger_eval_avg, yerr=hopper_dagger_eval_std, label='DAgger')
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Hopper Environment Imitation Learning')

plt.legend(loc='lower right')

fig.set_size_inches(10, 6)
#%%
ant_bc_train_avg = ant_dagger_train_avg
ant_bc_eval_avg = [4085.021240234375,4076.183349609375,4130.578125,\
                       3337.940185546875,4199.01025390625,3472.736328125,\
                       2410.2841796875,3800.73974609375,3292.074462890625]
ant_bc_eval_std = [479.2840270996094,230.29444885253906,17.60784149169922,\
                       906.5476684570312,148.285888671875,1452.61279296875,\
                       1607.117431640625,223.2168426513672,1127.9041748046875]

ant_bc_eval_avg_norm = np.array(ant_bc_eval_avg)/ant_bc_train_avg
ant_bc_eval_std_norm = np.array(ant_bc_eval_std)/ant_bc_train_avg

fig = plt.figure()
x = np.arange(2,11)

plt.errorbar(x, ant_bc_eval_avg, yerr=ant_bc_eval_std, label='Behavioral Cloning Agent')

plt.hlines(ant_dagger_train_avg,0,10,'r',label='Expert')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Average Return')
plt.title('Ant Behavioral Cloning with Varying Network Depth')


plt.legend(loc='lower right')

fig.set_size_inches(10, 6)


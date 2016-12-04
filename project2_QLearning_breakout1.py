# EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
# Information Theory & Machine Learning Lab, School of EE, KAIST
# Wonseok Jeon and Sungik Choi (wonsjeon@kaist.ac.kr, si_choi@kaist.ac.kr)
# written on 2016/11/23

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from project2_environment import environment1 as environment
from project2_state_representation import scalar_state

# First, set the parameters
num_rows = 5
num_columns = 5
num_states = 6**5
num_actions = 5
num_episodes = 10000
alpha = 0.1
gamma = 0.9
Q = np.zeros([num_states, num_actions])
num_trials = np.zeros([num_episodes])

for i_episode in range(num_episodes):
    S = [5, 5, 5, 5, 5]; Ss = scalar_state(S)
    epsilon = 1.0 - float(i_episode) / float(num_episodes - 1)
    for epoch in range(10000):
        if np.random.random() < epsilon:  # random exploration with prob. epsilon
            A = int(np.random.randint(0, 5))
        else:  # greedy action with random tie break
            maxQ = np.max(Q[Ss]) 
            A = int(np.random.choice(np.argwhere(Q[Ss] == maxQ).flatten()))
        R, Sn, T = environment(S, A)  # Observe the outputs of the state transition
        Q[Ss, A] = (1 - alpha) * Q[Ss, A] + alpha * (
            R + gamma * np.max(Q[scalar_state(Sn)]))  # Perform Q learning
        if T == 1:  # If terminal state
            num_trials[i_episode] = epoch + 1
            break
        S = Sn
# Print the number of trials
print "Number of time steps at the end of training: ", num_trials[num_episodes - 1]
# Plot the average number of time steps 
# Each data point is an average over (num_episodes / 100) episodes
Xaxis = np.linspace(1, 100, 100, endpoint = True)
C = np.mean(np.reshape(num_trials, [100, num_episodes / 100]), axis = 1)
plt.plot(Xaxis, C, '.')
plt.show()

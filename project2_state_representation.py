# EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
# Information Theory & Machine Learning Lab, School of EE, KAIST
# Wonseok Jeon and Sungik Choi (wonsjeon@kaist.ac.kr, si_choi@kaist.ac.kr)
# written on 2016/11/23

import numpy as np

def matrix_state(state):
    # This converts a length-5 state vector to 5x5 matrix
    # state: multiple instances (minibatch) of 5-dimensional vector
    # output: multiple instances of 5x5 matrix, each representing 5x5 bricks, where
    #     1 means a brick is present and 0 means no brick
    # Example
    #     state = [[5, 5, 5, 5, 5], [2, 5, 1, 3, 2]]
    #     output = [[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
    #              [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]]
    states_in_matrix = []
    for i in range(len(state)):
        states_in_matrix.append([])
        for j in range(len(state[i])):
            states_in_matrix[i].append([])
            if (state[i][j] < 6):
                for k in range(state[i][j]):
                    states_in_matrix[i][j].append(1)
                for k in range(state[i][j], 5):
                    states_in_matrix[i][j].append(0)
            if (state[i][j] == 6):
                for k in range(4):
                    states_in_matrix[i][j].append(0)
                for k in range(4, 5):
                    states_in_matrix[i][j].append(1)
    return states_in_matrix


def scalar_state(state):
    if min(state) == 0:
        states_in_scalar = 0 # Terminal state
    else:
        states_in_scalar = state[0]
        for i in range(1, len(state)):
            states_in_scalar = states_in_scalar * 5 + state[i]

    return states_in_scalar

# EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
# Information Theory & Machine Learning Lab, School of EE, KAIST
# Wonseok Jeon and Sungik Choi (wonsjeon@kaist.ac.kr, si_choi@kaist.ac.kr)
# written on 2016/11/23 

import numpy as np

def environment1(S, A):
    # Environment #1 for breakout game
    # If any one of the columns is cleared, all bricks are cleared
    if type(A).__module__ == np.__name__:   # for DQN   
        position = int(np.reshape(np.nonzero(A), [1]))
    else:           # for Q learning
        position = A
    Sn = list(S)    # copy array
    R = 0           # reward
    if Sn[position] > 0:
        Sn[position] -=  1
        # Break the bottom brick in the column corresponding to the action
        R = 1

    T = 0  # 1 for terminal state, 0 otherwise
    if min(Sn) == 0:
        T = 1  # End of the episode
        R += sum(Sn)  # Reward is the sum of all the remaining bricks
        for i in range(5):
            Sn[i]=0
    return R, Sn, T

def environment2(S, A):
    # Environment #2 for breakout game
    # If any one of the columns is cleared, all rows except the bottom row are cleared
    if type(A).__module__ == np.__name__:   # for DQN   
        position = int(np.reshape(np.nonzero(A), [1]))
    else:           # for Q learning
        position = A
    Sn = list(S)    # copy array
    R = 0
    T = 0
    if Sn[position] == 0:
        R = 0
    elif Sn[position] == 1:
        R = 1
        Sn[position] = 0
        for i in range(5):
            if Sn[i] <= 4:
                R += Sn[i]
                Sn[i] = 0
            elif Sn[i] == 5:
                R += 4
                Sn[i] = 6
            elif Sn[i] == 6:    # 6 means only the bottom brick is present
                Sn[i] = 6       # no change
    elif Sn[position] == 6:
        R = 1
        Sn[position] = 0
    elif Sn[position] >= 2 & Sn[position] <= 5:
        R = 1
        Sn[position] -= 1
    if max(Sn) == 0:
        T = 1
        for i in range(5):
            Sn[i]=0

    return R, Sn, T

def environment3(S, A):
    # Environment #3 for breakout game
    # If any two consecutive columns are cleared, all bricks are cleared
    if type(A).__module__ == np.__name__:   # for DQN   
        position = int(np.reshape(np.nonzero(A), [1]))
    else:           # for Q learning
        position = A
    Sn = list(S)    # copy array
    R = 0
    if Sn[position] > 0:
        Sn[position] -= 1
        R = 1

    T = 0
    if (Sn[0] + Sn[1]) * (Sn[1] + Sn[2]) * (Sn[2] + Sn[3]) * (Sn[3] + Sn[4]) == 0:
        R += sum(Sn)
        T = 1
        for i in range(5):
            Sn[i]=0

    return R, Sn, T

def environment_maze(S, A, M):
    # Given the state S, action A, and map M, determine reward, new_state, and terminal
    # Map is represented as a length-40 vector that represents the connectivity
    #   between state (0->1)(1->2)....(23->24),(0->5)(1->6)(2->7)....(19->24):
    #   1 means we can go through them and 0 means the tranition is blocked
    # Action: 0: Up, 1: Down, 2, Left 3: Right
    T = 0
    reward = 0
    k1 = int(S / 5)  # row
    k2 = S - 5 * k1  # column
    new_state = S
    if ((k1 == 4) and (A == 0)) or ((k1 == 0) and (A == 1)) or\
            ((k2 == 0) and (A == 2)) or ((k2 == 4) and (A == 3)):
        # invalid move
        new_state = S
    elif A == 0:  # Goes up
        if M[S + 20] == 0:  # if not blocked
            new_state = S + 5
    elif A == 1:
        if M[S + 20 - 5] == 0:
            new_state = S - 5
    elif A == 2:
        if M[4 * k1 + k2 - 1] == 0:
            new_state = S - 1
    elif A == 3:
        if M[4 * k1 + k2] == 0:
            new_state = S + 1
    if new_state == 24:
        reward = 1
        T = 1
    return reward, new_state, T

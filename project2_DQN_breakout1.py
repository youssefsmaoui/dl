# EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
# Information Theory & Machine Learning Lab, School of EE, KAIST
# Wonseok Jeon and Sungik Choi (wonsjeon@kaist.ac.kr, si_choi@kaist.ac.kr)
# written on 2016/11/23

import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from project2_environment import environment1 as environment
from project2_state_representation import matrix_state
#####################################################################
"""                    DEFINE HYPERPARAMETERS                     """
#####################################################################

# discount factor 
gamma = 0.9
# learning rate
alpha = 0.25
# size of minibatch
size_minibatch = 1
# whether to use experience replay or not
replay = 0
# number of episodes
num_episodes = 10
# period for updating target network
target_net_update_period = 1
# Performed time steps in each episdoe
num_trials=np.zeros([num_episodes])

#####################################################################
"""                COMPUTATIONAL GRAPH CONSTRUCTION               """
#####################################################################


### Q NETWORK ARCHITECTURES ###
def DeepQNetwork(state):
    # Set variable initializer.
    init = tf.random_normal_initializer()

    # Create variable named "weights1" and "biases1"
    weights1 = tf.get_variable("weights1", [5 * 5, 5], initializer=init)
    biases1 = tf.get_variable("biases1", [5], initializer=init)

    # Create 1st layer
    in1 = tf.reshape(state, [-1, 5 * 5])
    return tf.matmul(in1, weights1) + biases1


### SELECT Q NETWORK AND CORRESPONDING STATE DIMENSION.
QN, state_dim = DeepQNetwork, [None, 5, 5]
state_representation = matrix_state

### DATA FEED ###
S = tf.placeholder("float", shape=state_dim, name="Current_State")
A = tf.placeholder("float", shape=[None, 5], name="Action")
R = tf.placeholder("float", shape=[None, 1], name="Reward")
Sn = tf.placeholder("float", shape=state_dim, name="Next_State")
T = tf.placeholder("float", shape=[None, 1], name="Termination")

### ONLINE Q NETWORK ###
with tf.variable_scope("online_net"):
    online_Q = QN(S)

### TARGET Q NETWORK ###
with tf.variable_scope("target_net"):
    target_Q = QN(Sn)

### PARAMETERS ###
online_net_variables = tf.get_collection(
    tf.GraphKeys.VARIABLES, scope="online_net")
target_net_variables = tf.get_collection(
    tf.GraphKeys.VARIABLES, scope="target_net")

### LOSS FUNCTION ###
Y_targets = R + gamma * tf.mul(T, tf.reduce_max(target_Q, 1))
Y_onlines = tf.reduce_sum(tf.mul(online_Q, A), 1)
loss = tf.reduce_mean(tf.square(tf.sub(Y_targets, Y_onlines)))

### OPTIMIZER ###
optimizer = tf.train.RMSPropOptimizer(alpha).minimize(
    loss, var_list=online_net_variables)


### ONLINE NET UPDATER ###
def online_net_update(optimizer, minibatch):
    # Update parameters of online_net.
    sess.run(optimizer,
             feed_dict={
                 S: state_representation(minibatch[0]),
                 A: minibatch[1],
                 R: minibatch[2],
                 Sn: state_representation(minibatch[3]),
                 T: minibatch[4]
             })

    # Erase the contents in minibatch.
    return [[], [], [], [], []]


### TARGET NET UPDATER ###
target_net_update = [
    target_net_variables[i].assign(online_net_variables[i])
    for i in range(len(online_net_variables))
]

#####################################################################
"""                      TRAINING Q-NETWORK                       """
#####################################################################
### POSSIBLE ACTION LISTS ###
A_list = np.identity(5)
### EXPERIENCE REPLAY MEMORY (S, A, R, Sn, T) ###
replay_memory = [[], [], [], [], []]

### MINIBATCH FOR UPDATE (S, A, R, Sn, T) ###
minibatch = [[], [], [], [], []]
with tf.Session() as sess:
    ### VARIABLE INIIALIZATION ###
    sess.run(tf.initialize_all_variables())
    sess.run(target_net_update)

    ### COUNTER ###
    target_net_update_counter = 0

    ### TOTAL TIME STEP FOR EACH EPISODE ###
    tot_time_step = []
    time_episode_start = time.time()
    ### GAME PLAYING FOR num_episodes ### 
    for i_episode in range(num_episodes):

        # Epsilon for epsilon-greedy exploration
        epsilon = 1 - float(i_episode) / (num_episodes-1)

        # Check whether the episode is terminated or not
        T_ = 0

        # Initialize the time step for each episode.
        time_step = 0

        # Initialize the state.
        S_ = [5, 5, 5, 5, 5]

        # Check time of episode start.

        # Play the game until the end of the episode.
        while (not T_):
            # Check Q values
            Q_ = sess.run(online_Q, feed_dict={S: state_representation([S_])})
            # Select action based on epsilon-greedy strategy.
            if np.random.random() < epsilon:
                A_ = np.reshape(A_list[np.random.choice(5, 1)], [5])
                A_.astype(int)
            else:
                max_list = np.where(Q_[0] == max(Q_[0]))[0]
                A_ = A_list[np.random.choice(max_list)]

            # Get the next state and reward from the environment.
            R_, Sn_, T_ = environment(S_, A_)
            if replay:  # If the experience replay is used, 
                # Store data in experience replay memory.
                replay_memory[0].append(S_)
                replay_memory[1].append(A_)
                replay_memory[2].append([R_])
                replay_memory[3].append(Sn_)
                replay_memory[4].append([T_])

                # Size of the replay memory
                size_memory = len(replay_memory[0])

                # Sample uniformly random indices.
                num_sampling = min(size_memory, size_minibatch)
                sample_indices = np.random.randint(0, size_memory,
                                                   num_sampling)

                # Generate random minibatch.
                for i in range(len(replay_memory)):
                    minibatch[
                        i] = [replay_memory[i][j] for j in sample_indices]

                minibatch = online_net_update(optimizer, minibatch)

            else:  # If the experience replay is not used, 
                # Store data in minibatch.
                minibatch[0].append(S_)
                minibatch[1].append(A_)
                minibatch[2].append([R_])
                minibatch[3].append(Sn_)
                minibatch[4].append([T_])

                if len(minibatch[0]) == size_minibatch:  # If minibatch is full,
                    minibatch = online_net_update(optimizer, minibatch)
            
            # Apply state transition. 
            S_ = Sn_

            # Update parameters of target_net.
            if target_net_update_counter < target_net_update_period:
                target_net_update_counter += 1
            else:
                target_net_update_counter = 0
                sess.run(target_net_update)

            # Increase the time step
            time_step += 1

        # Check time of episode end.
        test_episode=1
        num_trials[i_episode]=time_step
        if i_episode % test_episode == test_episode-1:
            time_episode_end = time.time()
            print "Episode: ", i_episode+1, \
                "\t| Epsilon: ", "{0:.4f}".format(epsilon), \
                "\t| Final state :", S_, \
                "\t| Total time step : ", time_step
            print "Time: ", (time_episode_end - time_episode_start), "Sec"
            time_episode_start = time_episode_end
        # Report the total time step for this episode.
    tot_time_step.append(time_step)

# Print the number of trials
print "Number of time steps at the end of training: ", num_trials[num_episodes - 1]

print num_trials


# EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
# Information Theory & Machine Learning Lab (http://itml.kaist.ac.kr), School of EE, KAIST
# written by Jongmin Yoon
# 2016/11/08

import numpy as np
from numpy import random
import math
import tensorflow as tf
import h5py
from datetime import datetime
from crop_batch import crop_batch

# Variable setting
NUM_BATCHES = 32
NUM_EPOCH = 10

# Load HDF5 dataset and import as numpy array
train_filename = "./Caltech101_ten_train.h5"
val_filename = "./Caltech101_ten_val.h5"
h5f = h5py.File(train_filename, 'r+')
X_arr, Y_arr = h5f['/X'], h5f['/Y']
X, Y = np.array(X_arr), np.array(Y_arr)
h5f = h5py.File(val_filename, 'r+')
X_arr_val, Y_arr_val = h5f['/X'], h5f['/Y']
X_val, Y_val = np.array(X_arr_val), np.array(Y_arr_val)

output_weight_filename = './Caltech101_finetune_weight.npy'

# Import mean of ilsvrc2012 dataset
mean_filename = './ilsvrc_2012_mean.npy'
mean_data = np.load(mean_filename).mean(1).mean(1)
mean_data = mean_data.reshape((1, 1, 1, 3))
mean_val = np.repeat(mean_data, X_val.shape[0], axis=0)
mean_val = np.repeat(mean_val, 227, axis=1)
mean_val = np.repeat(mean_val, 227, axis=2)
mean_data = np.repeat(mean_data, NUM_BATCHES, axis=0)
mean_data = np.repeat(mean_data, 256, axis=1)
mean_data = np.repeat(mean_data, 256, axis=2)

# Internal variables for next_batch function
epochs_completed = 0
num_examples = X.shape[0]
index_in_epoch = num_examples + 1


# next_batch
def next_batch(batch_size):
    global X, Y
    global index_in_epoch, epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data has been used, it is reordered randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1

        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        X, Y = X[perm], Y[perm]

        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch

    X_crop = crop_batch(X[start:end], mean_data)
    return X_crop, Y[start:end]


# Make samples from validation set by making crops
def crop_val():
    global X_val, Y_val
    choose_crop = random.randint(low=0, high=30, size=(X_val.shape[0], 2))
    X_crop_val = np.zeros((X_val.shape[0], 227, 227, 3))

    for i in range(X_val.shape[0]):
        X_crop_val[i] = X_val[i, choose_crop[i, 0]:choose_crop[i, 0] + 227,
                              choose_crop[i, 1]:choose_crop[i, 1] + 227, :]

    X_crop_val = X_crop_val[:, :, :, [2, 1, 0]]
    X_crop_val *= 255.
    X_crop_val -= mean_val

    return X_crop_val, Y_val


Xb_val, Y_val = crop_val()

# Constructing AlexNet
net_data = np.load("./bvlc_alexnet.npy").item()
for x in net_data:
    exec ("%s = %s" % (str(x) + "W", "tf.Variable(net_data[x][0])"))
    exec ("%s = %s" % (str(x) + "b", "tf.Variable(net_data[x][1])"))


def conv(input,
         kernel,
         biases,
         k_h,
         k_w,
         c_o,
         s_h,
         s_w,
         padding="VALID",
         group=1):
    input_groups, kernel_groups = tf.split(3, group, input), tf.split(3, group,
                                                                      kernel)
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    output_groups = [
        convolve(i, k) for i, k in zip(input_groups, kernel_groups)
    ]
    conv = tf.concat(3, output_groups)
    return tf.reshape(
        tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


x = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))
y_ = tf.placeholder(tf.float32, shape=(None, 10))
cond = tf.placeholder(tf.int32, shape=[])

conv1 = tf.nn.relu(
    conv(
        x, conv1W, conv1b, 11, 11, 96, 4, 4, padding="VALID", group=1))
lrn1 = tf.nn.local_response_normalization(
    conv1, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
maxpool1 = tf.nn.max_pool(
    lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

conv2 = tf.nn.relu(
    conv(
        maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2))
lrn2 = tf.nn.local_response_normalization(
    conv2, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
maxpool2 = tf.nn.max_pool(
    lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

conv3 = tf.nn.relu(
    conv(
        maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1))
conv4 = tf.nn.relu(
    conv(
        conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2))
conv5 = tf.nn.relu(
    conv(
        conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2))
maxpool5 = tf.nn.max_pool(
    conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

fc6 = tf.nn.relu_layer(
    tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W,
    fc6b)
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

fan1 = math.sqrt(6.0 / (4096.0 + 10.0))
fc8W = tf.Variable(tf.random_uniform([4096, 10], minval=-fan1, maxval=fan1))
fc8b = tf.Variable(tf.constant(0.0, shape=[10]))
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

y_conv = tf.nn.softmax(fc8)

y_reshape = tf.cond(cond > 0,
                    lambda: tf.reshape(y_conv, [NUM_BATCHES, 10, 10]),
                    lambda: y_conv)
y_fin = tf.cond(cond>0, lambda: tf.reduce_mean(y_reshape, reduction_indices=1), lambda: y_conv)

# Train and evaluate the model
temp_y = (y_ * tf.log(y_fin))
cross_entropy = -tf.reduce_sum(temp_y)
lr = tf.train.exponential_decay(learning_rate=1e-4, global_step=tf.Variable(0, trainable=False), \
        decay_steps=num_examples//NUM_BATCHES, decay_rate=0.95, staircase=True)
opt = tf.train.MomentumOptimizer(lr, momentum=0.0)
train_step = opt.minimize(cross_entropy, var_list=[fc8W, fc8b])
correct_prediction = tf.equal(tf.argmax(y_fin, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
correct_2 = tf.nn.in_top_k(y_fin, tf.argmax(y_, 1), 2)
acc_2 = tf.reduce_mean(tf.cast(correct_2, tf.float32))
correct_3 = tf.nn.in_top_k(y_fin, tf.argmax(y_, 1), 3)
acc_3 = tf.reduce_mean(tf.cast(correct_3, tf.float32))

acc_t = tf.pack([acc, acc_2, acc_3])

# Saver
saver = tf.train.Saver({'fc8W': fc8W, 'fc8b': fc8b})

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

print(
    "================================================================================="
)
print("|Time\t\t\t\t|Epoch\t|Batch\t|Set\t|Top-1\t|Top-2\t|Top-3\t|")
print(
    "|===============================================================================|"
)

for i in range(NUM_EPOCH):
    for j in range(num_examples // NUM_BATCHES):
        Xb, Yb = next_batch(NUM_BATCHES)
        #acc_print=y_.eval(feed_dict={x:Xb,y_:Yb,cond:1},session=sess)
        train_acc = acc_t.eval(
            feed_dict={x: Xb,
                       y_: Yb,
                       cond: 1}, session=sess)
        print("|" + str(datetime.now()) +
              "\t|%d\t|%d\t|train\t|%4.4f\t|%4.4f\t|%4.4f\t|" % (
                  i + 1, j + 1, train_acc[0], train_acc[1], train_acc[2]))
        sess.run(train_step, feed_dict={x: Xb, y_: Yb, cond: 1})
    test_acc = acc_t.eval(
        feed_dict={x: Xb_val,
                   y_: Y_val,
                   cond: -1}, session=sess)
    print("|" + str(datetime.now()) +
          "\t|%d\t|-\t|test\t|%4.4f\t|%4.4f\t|%4.4f\t|" % (i + 1, test_acc[
              0], test_acc[1], test_acc[2]))
    npy_save = {}
    npy_save[0] = fc8W.eval()
    npy_save[1] = fc8b.eval()
    np.save(output_weight_filename, npy_save)

save_path = saver.save(sess, "./finetune_fc.ckpt")
print("Fine-tuned network saved in file: %s" % save_path)

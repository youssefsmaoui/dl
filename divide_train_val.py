from __future__ import print_function

import random
import os


def divide_train_val():
    input_file = open('./101_labels_ten.txt', 'r')
    output_train_file = open('./project2_train_label.txt', 'w')
    output_val_file = open('./project2_val_label.txt', 'w')

    val_ratio = 0.2

    random.seed()

    label_list = input_file.readlines()
    for l in label_list:
        randnum = random.random()
        if randnum < 0.2:
            output_val_file.write(l)
        else:
            output_train_file.write(l)

    input_file.close()
    output_train_file.close()
    output_val_file.close()

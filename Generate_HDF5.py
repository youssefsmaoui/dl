# Import
from __future__ import division, print_function, absolute_import

import os
import h5py
import random
import numpy as np
from PIL import Image
from datetime import datetime
import tflearn.data_utils as data_utils


def build_hdf5_image_dataset_simp(target_path,
                                  output_path,
                                  image_shape=(256, 256)):
    with open(target_path, 'r') as f:
        print("Start reading target path file...")
        images, labels = [], []
        for l in f.readlines():
            l = l.strip('\n').split()
            images.append(l[0])
            labels.append(int(l[1]))
        print("Finished reading target path file!")

    print("Number of images is " + str(len(images)))

    n_classes = np.max(labels) + 1
    print("Number of classes is " + str(n_classes))

    size_image = len(images)
    img_sum = np.zeros((256, 256, 3))

    d_imgshape = (size_image, image_shape[0], image_shape[1], 3)
    d_labelshape = (size_image, n_classes)

    dataset = h5py.File(output_path, 'w')
    dataset.create_dataset('X', d_imgshape, chunks=None)
    dataset.create_dataset('Y', d_labelshape, chunks=None)

    for i in range(size_image):
        if i % 100 == 0:
            print(str(datetime.now()) + "\t" + str(i))

        # PIL type
        im1 = Image.open(images[i])
        if im1.mode != 'RGB':
            im1 = im1.convert('RGB')

        im1 = im1.resize((256, 256), Image.ANTIALIAS)
        img = np.asarray(im1, dtype='float32')
        img /= 255.

        # Export
        dataset['X'][i] = img
        dataset['Y'][i] = data_utils.to_categorical([labels[i]], n_classes)[0]

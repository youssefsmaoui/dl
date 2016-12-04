from Generate_HDF5 import build_hdf5_image_dataset_simp
from divide_train_val import divide_train_val

# Divide images into train and validation sets
divide_train_val()

# Training set generation
build_hdf5_image_dataset_simp(
    target_path='./project2_train_label.txt',
    output_path='./Caltech101_ten_train.h5')

# Test set generation
build_hdf5_image_dataset_simp(
    target_path='./project2_val_label.txt',
    output_path='./Caltech101_ten_val.h5')

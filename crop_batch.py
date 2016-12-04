import numpy as np

crop_entry = [[0, 0], [0, 29], [29, 0], [29, 29], [14, 14]]

def crop_batch(X, mean_data):
    X = X[:, :, :, [2, 1, 0]]
    X *= 255.
    X -= mean_data
    X_crop = np.empty((10 * X.shape[0], 227, 227, 3), dtype=np.float32)
    for k in range(X.shape[0]):
        for l in range(5):
            X_crop[k * 10 + l, :, :, :] = X[k, crop_entry[l][0]:crop_entry[l][
                0] + 227, crop_entry[l][1]:crop_entry[l][1] +
                                            227, :]  
        X_crop[k * 10 + 5:k * 10 + 10, :, :, :] = X_crop[
            k * 10:k * 10 + 5, :, ::-1, :]  

    return X_crop

import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical


def unpickle_file(file_path):
    with open(file_path, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def create_dataset(list_file_path):
    list_data = []
    list_label = []
    for file_path in list_file_path:
        file_path = 'data/' + file_path
        data = unpickle_file(file_path)
        x = data[b'data']
        y = data[b'labels']
        tmp = []
        for i in range(len(x)):
            r_c = x[i][:1024].reshape((32, 32))
            g_c = x[i][1024:2048].reshape((32, 32))
            b_c = x[i][2048:].reshape((32, 32))
            img = np.dstack((r_c, g_c, b_c))
            tmp.append(img)
        x = np.array(tmp)
        list_data.append(x)
        list_label.append(to_categorical(y, 10))

    return np.concatenate(tuple(list_data)), np.concatenate(tuple(list_label))



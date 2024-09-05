import numpy as np
import pandas as pd
import sys
import pickle

from os import listdir
from os.path import isfile, join


# import from MNIST dataset on AWS using the instructions here: https://stackoverflow.com/a/40693405/4147579
def unpickle():
    file_path = "./mnist.pkl.gz"
    import gzip

    f = gzip.open(file_path, "rb")
    if sys.version_info < (3,):
        # TODO: pyright is telling me this is unreachable, check later
        data = pickle.load(f)
    else:
        data = pickle.load(f, encoding="bytes")
    f.close()
    return data


def process_labels(labels):
    label_to_binary = np.zeros([10, 4], dtype="int8")
    for i in np.arange(10):
        label_to_binary[i] = np.array(list(np.binary_repr(i, 4)), dtype=int)

    # Changing the labels from 0-9 int to binary for our weightless neural state machine
    labels_z = np.zeros([labels.size, 4])
    for i in np.arange(labels.size):
        labels_z[i] = label_to_binary[labels[i]]
    return labels_z


def process_data():
    data = unpickle()
    (MN_TRAIN, MN_TRAIN_labels), (MN_TEST, MN_TEST_labels) = data
    MN_TRAIN_Z = process_labels(MN_TRAIN_labels)
    MN_TEST_Z = process_labels(MN_TEST_labels)
    return (MN_TRAIN, MN_TRAIN_Z), (MN_TEST, MN_TEST_Z)


def random_sample(num_samples, samples, labels):
    index = np.random.choice(samples.shape[0], num_samples, replace=False)
    return samples[index], labels[index]


def down_sample_item(x, down=200):
    f = np.vectorize(lambda x, down: 1 if x >= down else 0)
    return f(x, down)
# def down_sample(image, down=200):
#     down_image = np.zeros(image.shape)
#     down_image[image < down] = 0
#     down_image[image >= down] = 1
#     return down_image


def get_font_data(filename):
    df = pd.read_excel(filename)
    df = df.iloc[:, :145]
    arr = df.to_numpy()
    arr = np.append(arr[:28], arr[30:58], axis=1)
    arr = np.delete(arr, [28 + 29 * i for i in range(10)], axis=1)
    ret_arr = np.zeros((10, 28, 28), dtype=np.uint8)
    for i in range(10):
        for r in range(28):
            for c in range(28):
                # TODO: python is complaining about some cells being cast from NaN,
                #       but it seems to work anyways. Find a fix for this
                ret_arr[i][r][c] = 255 * arr[r][c + 28 * i]
    label_to_binary = np.zeros([10, 4], dtype="int8")
    for i in np.arange(10):
        label_to_binary[i] = np.array(list(np.binary_repr(i, 4)), dtype=int)
    return (ret_arr, np.array([label_to_binary[i] for i in range(10)]))


def get_all_fonts():
    folder = "./Fonts/"
    items = listdir(folder)
    fonts = {}
    for item in items:
        # print(item)
        if not isfile(join(folder, item)):
            continue
        file = join(folder, item)
        fonts[item.removesuffix(".xlsx")] = get_font_data(file)
    return fonts


def select_training_fonts(fonts):
    inputs = []
    outputs = []
    for f in fonts:
        if f == "MNIST":
            continue
        inputs.extend(FONTS[f][0])
        outputs.extend(FONTS[f][1])
    return np.array(inputs), np.array(outputs)


(MN_TRAIN, MN_TRAIN_Z), (MN_TEST, MN_TEST_Z) = process_data()
FONTS = get_all_fonts()

import numpy as np
import pandas as pd
import sys
import pickle

from os import listdir
from os.path import isfile, join
from scipy.signal import convolve2d


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


def convolve(image, kernel):
    if image.ndim == 2:  # Single image
        return convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
    
    elif image.ndim == 3:  # many images at once
        return np.array([convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0) for img in image])
    
    else:
        raise ValueError("Invalid image shape.")

gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]) / 16.0

def max_pooling(image, pool_size=2, stride=2):
    def pool_single_image(img):
        h, w = img.shape
        new_h = int(np.ceil(h / stride))  
        new_w = int(np.ceil(w / stride))

        pooled_image = np.zeros((new_h, new_w))

        for i in range(new_h):
            for j in range(new_w):
                x_start, y_start = i * stride, j * stride
                x_end, y_end = min(x_start + pool_size, h), min(y_start + pool_size, w)  
                pooled_image[i, j] = np.max(img[x_start:x_end, y_start:y_end])

        return pooled_image

    if image.ndim == 2:  # Single image 
        return pool_single_image(image)
    
    elif image.ndim == 3:  # many images at once
        return np.array([pool_single_image(img) for img in image])
    
    else:
        raise ValueError(f"Invalid image shape. ")


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


def random_sample(num_samples, samples, labels, seed=42):
    np.random.seed(seed)
    index = np.random.choice(samples.shape[0], num_samples, replace=False)
    return samples[index], labels[index]


def down_sample_item(x, down=200):
    f = np.vectorize(lambda x, down: 1 if x >= down else 0)
    return f(x, down)

def down_sample(image, down=150):
    down_image = np.zeros(image.shape)
    down_image[image < down] = 0
    down_image[image >= down] = 1
    return down_image

def bitmap_to_binary(image):
    if image.ndim == 1:
        # Handle 1D array
        return np.array([np.array(list(format(pixel, '08b')), dtype=np.uint8) for pixel in image])
    # elif image.ndim == 2:
    #     # Handle 2D array
    #     return np.array([[np.array(list(format(pixel, '08b')), dtype=np.uint8) for pixel in row] for row in image])
    else:
        # Handle 3D or higher-dimensional arrays
        return np.array([bitmap_to_binary(sub_array) for sub_array in image])

def get_font_data(filename):
    df = pd.read_excel(filename)
    # cut off dataframe at end of valid columns
    df = df.iloc[:, :145]
    arr = df.to_numpy()
    # spreadsheet has two layers of digits
    # this moves them into the same layer
    arr = np.append(arr[:28], arr[30:58], axis=1)
    arr = np.delete(arr, [28 + 29 * i for i in range(10)], axis=1)
    ret_arr = np.zeros((10, 28, 28), dtype=np.uint8)
    for i in range(10):
        for r in range(28):
            for c in range(28):
                # TODO: python is complaining about some cells being cast from NaN,
                #       but it seems to work anyways. Find a fix for this

                # Multiplying by 255 so it downsamples correctly
                # print("file: {}, digit: {}, r:, {}, c: {}".format(filename, i, r, c))
                ret_arr[i][r][c] = 255 * arr[r][c + 28 * i]
    # Convert labels
    label_to_binary = np.zeros([10, 4], dtype="int8")
    for i in np.arange(10):
        label_to_binary[i] = np.array(list(np.binary_repr(i, 4)), dtype=int)

    return (ret_arr, np.array([label_to_binary[i] for i in range(10)]))


def get_all_fonts():
    folder = "./Fonts/"
    items = listdir(folder)
    fonts = {}
    for item in items:
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

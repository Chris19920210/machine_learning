import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate
from math import log
from skimage import transform as tf
from skimage import io
import re
import os


def calibrate(pic_array, target_shape):
    # resize picture to the target_shape
    # pic_array is a numpy nd array of the picture with range(0, 255)
    # target_shape is a tuple, eg. (100, 100)

    width, length = pic_array.shape
    if width > length:
        pad_length = width - length
        half_width = int(pad_length / 2)
        new_pic_array = np.lib.pad(pic_array,
                                   ((0, 0), (half_width, pad_length - half_width)),
                                   'constant', constant_values=255)
    elif length > width:
        pad_length = length - width
        half_width = int(pad_length / 2)
        new_pic_array = np.lib.pad(pic_array,
                                   ((half_width, pad_length - half_width), (0, 0)),
                                   'constant', constant_values=255)
    else:
        new_pic_array = pic_array

    assert(new_pic_array.shape[0] == new_pic_array.shape[1])

    pic_array_resized = resize(new_pic_array, target_shape)

    return pic_array_resized


def random_trans_single_output(pic_array):
    # randomly transform the pic_array, which is a numpy nd array
    # flipping
    do_hori_flip = np.random.binomial(1, 0.5)
    if do_hori_flip:
        pic_array = np.fliplr(pic_array)

    do_vert_flip = np.random.binomial(1, 0.5)
    if do_vert_flip:
        pic_array = np.flipud(pic_array)

    # rotation
    pic_array = rotate(pic_array, np.random.random_integers(0, 360),
                       mode='constant', cval=1)

    # scaling
    scale_ratio = log(np.random.uniform(2.5, 4.5))
    afine_tf = tf.AffineTransform(scale=(scale_ratio, scale_ratio))
    pic_array = tf.warp(pic_array, afine_tf, mode='constant', cval=1)

    # translation
    trans_length = np.random.random_integers(-6, 6, 2)
    trans_length = (trans_length[0], trans_length[1])
    afine_tf = tf.AffineTransform(translation=trans_length)
    pic_array = tf.warp(pic_array, afine_tf, mode='constant', cval=1)

    return pic_array


def generate_pics(pic_array, n):
    # randomly transform n pictures based on pic_array
    # return a list of pictures
    pictures = []
    for i in range(n):
        pictures.append(random_trans_single_output(pic_array))
    return pictures


# read selected class_set
class_set = []
with open("../Data/selected_classes.txt", 'r') as f:
    for line in f:
        class_set.append(line.strip())
print(class_set)
class_map = {class_set[k]: k for k in range(len(class_set))}

# Data is the path to all images
# y is the response label
Data = []
y = []

# number of replications in the enlargement process
reps = 10

walk_dir = '/home/s_ariel/Documents/Kaggle/Data/train'
for root, subdirs, files in os.walk(walk_dir):
    if os.path.basename(root) in class_set:
        for file in files:
            Data.append(root + '/' + file)
            y += [os.path.basename(root)] * reps

y = map(lambda x: class_map[x], y)
y = np.asarray(list(y))

# read and flatten picture data into X
target_shape = (32, 32)
X = []
n = len(Data)
for i in range(5):
    print(i, Data[i])
    pic_array = io.imread(Data[i], as_grey=True)
    pic_array = calibrate(pic_array, target_shape)
    pictures = generate_pics(pic_array, reps)
    pictures = map(np.ravel, pictures)
    X += pictures

X = np.asarray(X, dtype='float32')
print(X.shape, y.shape)
y = y[:50]
data = np.column_stack((y, X))
# np.savetxt("../Data/selected_dataset.txt", data)


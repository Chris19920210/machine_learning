"""
This file load the jpg from training set and test set respectively
Do the calibrate(resize) and shuffling
The final output would be 4 numpy ndarrays, train_X, train_y, test_X, test_y
the data type for training set is uint8, the data type for test set is int(0,1,...,7)
"""

import gzip, pickle
from skimage.transform import resize, rotate
import numpy as np
import os
from skimage import io
import random
from sklearn.cross_validation import StratifiedShuffleSplit
"""
Do not forget this step if it is running on a mac os
# command to delete .DS_Store in cd
find . -name '*.DS_Store' -type f -delete
"""


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

    pic_array_resized = resize(new_pic_array, target_shape,
                               preserve_range=True, mode='constant',
                               cval=255, order=1)
    pic_array_resized = np.asarray(pic_array_resized, dtype='uint8')

    return pic_array_resized



# class_set and class_map transfer all labels to 0,1,2,..,7
class_set = []
path = '/home/chris/train'
dirs = os.listdir(path)

for fil in dirs:
    class_set.append(fil)

class_map = {class_set[k]: k for k in range(len(class_set))}
pickle.dump(class_map, open( "/home/chris/class.p", "wb" ) )



"""
Do the transformation for training set
The output should be a ndarray(#training sample,1,32,32) train_X
and a ndarray(#training sample,1) train_y. Both train_X and train_y should be shuffled

"""

# Train_Data contains the picture name, y contains its corresponding label
Data = []
y = []
walk_dir = path
for root, subdirs, files in os.walk(walk_dir):
        for file in files:
            Data.append(root + '/' + file)
            y += [os.path.basename(root)]

y = map(lambda x: class_map[x], y)
y = np.asarray(list(y), dtype='uint8')

n = len(Data)
# X.shape is (n,1,32,32) is ndarray. First coordinate is the id of picture
# Second coordinate is the channels of image
# 3rd and 4th argument is the pixels of that image, range is (0,255)
# y is also ndarray that that has shape (n,1)
target_shape = (100, 100)
X = np.ndarray([n, 1, 100, 100],dtype='uint8')
for i in range(n):
    pic_array = io.imread(Data[i], as_grey=True)
    pic_array = calibrate(pic_array, target_shape)
    X[i, 0] = pic_array

sss = StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=0)

for train_index, test_index in sss:
    train_X, test_X = X[train_index], X[test_index]
    train_y, test_y = y[train_index], y[test_index]


"""
Store the value into file.pkl.gz
"""

train_set = train_X, train_y
test_set = test_X, test_y
dataset = [train_set, test_set]



np.save('/home/chris/dataset_wholeset', dataset)


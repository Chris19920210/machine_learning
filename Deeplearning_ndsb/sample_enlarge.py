import os
import sys
from pic_preprocess import makeThumb, zoom
from PIL import Image, ImageChops, ImageOps
import numpy as np
from skimage import data
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

walk_dir = '/home/s_ariel/Documents/Kaggle/Data/train'

# for root, subdirs, files in os.walk(walk_dir):
#     for file in files:
#         print(file)
#         image_path = root + '/' + file
#         im = Image.open(image_path)
#         im_arr = np.asarray(im)
#         max_size = max(im_arr.shape)
#         im_arr_zoom = makeThumb(image_path, (max_size, max_size))
#         im_arerr_zoom2 = resize(im_arr_zoom, (100, 100))
#         im_arr_zoom2 = im_arr_zoom2[:, :, 0]
#         np.savetxt(root + '/' + file + ".txt", im_arr_zoom2)


# for root, subdirs, files in os.walk(walk_dir):
#     for dir in subdirs:
#         print(dir)
#         os.system("/bin/ls " + root + '/' + dir + " | /bin/grep -c txt")

class_set = []
with open("../Data/selected_classes.txt", 'r') as f:
    for line in f:
        class_set.append(line.strip())

Data = []
y = []
match_txt = re.compile(".*\.txt")
for root, subdirs, files in os.walk(walk_dir):
    for subdir in subdirs:
        if subdir in class_set:
            for file in os.listdir(root + '/' + subdir):
                if match_txt.match(file):
                    Data.append(root + '/' + subdir + '/' + file)
                    y.append(subdir)

y = np.asarray(y)
print(y)

n = len(Data)
X = np.zeros((n, 100 * 100), dtype="float32")
for i in range(n):
    print(i)
    data_tmp = np.genfromtxt(Data[i])
    X[i, :] = data_tmp.ravel()

np.save("../Data/X_selected", X)
np.save("../Data/y_selected", y)

X = np.load("../Data/X_selected.npy")
y = np.load("../Data/y_selected.npy")
print(X[7344, :])
print(y.shape)



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


path = '/home/chris/machine_learning_final_project/keras/'
train, test = np.load(path + 'dataset.npy')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

"""
f = gzip.open('/home/ubuntu/ndsb/keras/file.pkl.gz', 'rb')
train, test = Cpickle.load(f)
f.close()
"""
# the data, shuffled and split between tran and test sets
X_train,y_train=train
X_test,y_test=test

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

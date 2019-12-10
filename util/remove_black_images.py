import os
import cv2
import numpy

dir = "C:/Users/jeffp/pix2pix/datasets/cervoai_pix2pix"
dir_train = dir + "/train"
dir_test = dir + "/test"
dir_val = dir + "/val"
dir_list = [dir_train, dir_test, dir_val]
for directory in dir_list:
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            img = cv2.imread(subdir + "/" + file)
            if numpy.sum(img) == 0:
                print(file + " does not contain an image and will be removed.")
                os.remove(subdir + "/" + file)

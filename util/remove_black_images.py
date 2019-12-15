import os
import cv2
import numpy

dir = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/datasets/cervoai_pix2pix_axial"
dir_train = dir + "/train"
dir_test = dir + "/test"
dir_val = dir + "/val"
dir_test_fail = dir + "/test_fail"
dir_list = [dir_train, dir_test, dir_val]
for directory in dir_list:
    breakpoint()
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            img = cv2.imread(subdir + "/" + file)
            if numpy.sum(img) == 0:
                print(subdir + "/" + file + " does not contain an image and will be removed.")
                os.remove(subdir + "/" + file)

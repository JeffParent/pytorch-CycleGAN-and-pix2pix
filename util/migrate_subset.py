import os
import shutil

dir = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/datasets/cervoai_pix2pix"

dir_train = dir + "/train"
dir_test = dir + "/test"
dir_val = dir + "/val"

dir_list = [dir_train, dir_test, dir_val]
images = ('_7.png', '_8.png', '_9.png', '_10.png', '_11.png', '_12.png', '_13.png')
for directory in dir_list:
    i = 0
    print("Directory: " + directory)
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            print(str(i))
            i += 1
            if file.endswith(images):
                shutil.copy(subdir.replace("\\", "/") + "/" + file, directory + "_subset")

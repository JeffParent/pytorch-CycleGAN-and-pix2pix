import os
import cv2
import numpy
import imagehash
from PIL import Image, ImageChops

dir_pass = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest_pass/images"
dir_fail = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest/images"
dir_pass_diff = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest_pass/images_diff"
dir_fail_diff = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest/images_diff"

def mse(imageA, imageB):
    err = numpy.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

results_mse = []
results_diff = []
results_diff_fake_ratio = []
pass_brains = {}
for subdir, dirs, files in os.walk(dir_pass):
    for file in files:
        if file.split("_")[0] not in pass_brains.keys():
            pass_brains[file.split("_")[0]] = 0
        if file.endswith("real_B.png"):
            img_real_mse = cv2.imread(dir_pass + "/" + file)
            img_fake_mse = cv2.imread(dir_pass + "/" + file.replace("real", "fake"))
            img_real_diff = Image.open(dir_pass + "/" + file)
            img_fake_diff = Image.open(dir_pass + "/" + file.replace("real", "fake"))
            im1 = imagehash.average_hash(img_real_diff, hash_size=256)
            im2 = imagehash.average_hash(img_fake_diff, hash_size=256)
            pass_brains[file.split("_")[0]] = pass_brains[file.split("_")[0]] + (abs(im1 - im2))

print("Diff Pass Means: " + str(numpy.array(list(pass_brains.values())).mean()))

fail_brains = {}
results_mse = []
results_diff = []
results_diff_fake_ratio = []
results_diff_real_ratio = []
for subdir, dirs, files in os.walk(dir_fail):
    for file in files:
        if file.split("_")[0] not in fail_brains.keys():
            fail_brains[file.split("_")[0]] = 0
        if file.endswith("real_B.png"):
            img_real_mse = cv2.imread(dir_fail + "/" + file)
            img_fake_mse = cv2.imread(dir_fail + "/" + file.replace("real", "fake"))
            img_real_diff = Image.open(dir_fail + "/" + file)
            img_fake_diff = Image.open(dir_fail + "/" + file.replace("real", "fake"))
            im1 = imagehash.average_hash(img_real_diff, hash_size=256)
            im2 = imagehash.average_hash(img_fake_diff, hash_size=256)
            fail_brains[file.split("_")[0]] = fail_brains[file.split("_")[0]] + (abs(im1 - im2))

print("Diff Fail Means: " + str(numpy.array(list(fail_brains.values())).mean()))

# len([k for k,v in pass_brains.items() if float(v) <= 6150])
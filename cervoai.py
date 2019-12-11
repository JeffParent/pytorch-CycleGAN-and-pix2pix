import os
import cv2
import numpy
from PIL import Image, ImageChops

dir_pass = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest_pass/images"
dir_fail = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest/images"


def mse(imageA, imageB):
    err = numpy.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

results_mse = []
results_diff = []
for subdir, dirs, files in os.walk(dir_pass):
    for file in files:
        if file.endswith("real_B.png"):
            img_real_mse = cv2.imread(dir_pass + "/" + file)
            img_fake_mse = cv2.imread(dir_pass + "/" + file.replace("real", "fake"))
            img_real_diff = Image.open(dir_pass + "/" + file)
            img_fake_diff = Image.open(dir_pass + "/" + file.replace("real", "fake"))
            result_mse = mse(img_real_mse, img_fake_mse)
            result_diff = ImageChops.difference(img_real_diff, img_fake_diff)
            result_diff = numpy.sum(numpy.where(numpy.asarray(result_diff) > 0, 1, 0))
            results_mse.append(result_mse)
            results_diff.append(result_diff)
print("MSE Pass Means: " + str(numpy.mean(results_mse)))
print("MSE Pass Min: " + str(numpy.min(results_mse)))
print("MSE Pass Max: " + str(numpy.max(results_mse)))

print("Diff Pass Means: " + str(numpy.mean(results_diff)))
print("Diff Pass Min: " + str(numpy.min(results_diff)))
print("Diff Pass Max: " + str(numpy.max(results_diff)))

results_mse = []
results_diff = []
for subdir, dirs, files in os.walk(dir_fail):
    for file in files:
        if file.endswith("real_B.png"):
            img_real_mse = cv2.imread(dir_fail + "/" + file)
            img_fake_mse = cv2.imread(dir_fail + "/" + file.replace("real", "fake"))
            img_real_diff = Image.open(dir_fail + "/" + file)
            img_fake_diff = Image.open(dir_fail + "/" + file.replace("real", "fake"))
            result_mse = mse(img_real_mse, img_fake_mse)
            result_diff = ImageChops.difference(img_real_diff, img_fake_diff)
            result_diff = numpy.sum(numpy.where(numpy.asarray(result_diff) > 0, 1, 0))
            results_mse.append(result_mse)
            results_diff.append(result_diff)
print("MSE Fail Means: " + str(numpy.mean(results_mse)))
print("MSE Fail Min: " + str(numpy.min(results_mse)))
print("MSE Fail Max: " + str(numpy.max(results_mse)))

print("Diff Fail Means: " + str(numpy.mean(results_diff)))
print("Diff Fail Min: " + str(numpy.min(results_diff)))
print("Diff Fail Max: " + str(numpy.max(results_diff)))

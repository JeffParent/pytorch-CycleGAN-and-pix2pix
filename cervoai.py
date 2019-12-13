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
# results_diff_real_ratio = []
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
            # img_real_diff_sum = numpy.sum(numpy.where(numpy.asarray(img_real_diff) > 0, 1, 0))
            # breakpoint()
            # result_mse = mse(img_real_mse, img_fake_mse)
            # result_diff = ImageChops.difference(img_real_diff, img_fake_diff)
            # result_diff.save(dir_pass_diff + "/" + file.replace("real_B", "diff"), "PNG")
            # result_diff = numpy.sum(numpy.where(numpy.asarray(result_diff) > 0, 1, 0))
            # result_diff_sum = 0
            # result_diff_total = 0
            # for pixels in numpy.asarray(result_diff):
            #     for pixel in pixels:
            #         # result_diff_total += 1
            #         if numpy.sum(pixel) != 0:
            #             result_diff_sum += 1

            # results_mse.append(result_mse)
            # results_diff.append(result_diff_sum)
            # results_diff_fake_ratio.append(result_diff_sum/img_fake_diff_sum)
            # results_diff_real_ratio.append(result_diff / img_real_diff_sum)

# print("MSE Pass Means: " + str(numpy.mean(results_mse)))
# print("MSE Pass Min: " + str(numpy.min(results_mse)))
# print("MSE Pass Max: " + str(numpy.max(results_mse)))

print("Diff Pass Means: " + str(numpy.array(list(pass_brains.values())).mean()))
# print("Diff Pass Min: " + str(numpy.min(brains.values())))
# print("Diff Pass Max: " + str(numpy.max(brains.values())))
#
# print("Diff Fake Ratio Pass Means: " + str(numpy.mean(results_diff_fake_ratio)))
# print("Diff Fake Ratio Pass Min: " + str(numpy.min(results_diff_fake_ratio)))
# print("Diff Fake Ratio Pass Max: " + str(numpy.max(results_diff_fake_ratio)))

# print("Diff Real Ratio Pass Means: " + str(numpy.mean(results_diff_real_ratio)))
# print("Diff Real Ratio Pass Min: " + str(numpy.min(results_diff_real_ratio)))
# print("Diff Real Ratio Pass Max: " + str(numpy.max(results_diff_real_ratio)))
#
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
            # img_fake_diff_sum = numpy.sum(numpy.where(numpy.asarray(img_fake_diff) > 0, 1, 0))
            # img_real_diff_sum = numpy.sum(numpy.where(numpy.asarray(img_real_diff) > 0, 1, 0))
#
#             result_mse = mse(img_real_mse, img_fake_mse)
#             result_diff = ImageChops.difference(img_real_diff, img_fake_diff)
#             # result_diff.save(dir_fail_diff + "/" + file.replace("real_B", "diff"), "PNG")
#             result_diff = numpy.sum(numpy.where(numpy.asarray(result_diff) > 0, 1, 0))
#             results_mse.append(result_mse)
#             results_diff.append(result_diff)
#             results_diff_fake_ratio.append(result_diff / img_fake_diff_sum)
#             results_diff_real_ratio.append(result_diff / img_real_diff_sum)
#
# print("MSE Fail Means: " + str(numpy.mean(results_mse)))
# print("MSE Fail Min: " + str(numpy.min(results_mse)))
# print("MSE Fail Max: " + str(numpy.max(results_mse)))
print("Diff Fail Means: " + str(numpy.array(list(fail_brains.values())).mean()))
breakpoint()
# print("Diff Fail Means: " + str(numpy.mean(results_diff)))
# print("Diff Fail Min: " + str(numpy.min(results_diff)))
# print("Diff Fail Max: " + str(numpy.max(results_diff)))
#
# print("Diff Ratio Fail Means: " + str(numpy.mean(results_diff_fake_ratio)))
# print("Diff Ratio Fail Min: " + str(numpy.min(results_diff_fake_ratio)))
# print("Diff Ratio Fail Max: " + str(numpy.max(results_diff_fake_ratio)))
#
# print("Diff Real Ratio Fail Means: " + str(numpy.mean(results_diff_real_ratio)))
# print("Diff Real Ratio Fail Min: " + str(numpy.min(results_diff_real_ratio)))
# print("Diff Real Ratio Fail Max: " + str(numpy.max(results_diff_real_ratio)))
# len([k for k,v in pass_brains.items() if float(v) <= 5500])
import os
import numpy
import imagehash
from PIL import Image
from matplotlib import pyplot

dir_pass = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest_pass/images"
dir_fail = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest/images"
dir_pass_diff = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest_pass/images_diff"
dir_fail_diff = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest/images_diff"

images = ('_7_real_B.png', '_8_real_B.png', '_9_real_B.png', '_10_real_B.png', '_11_real_B.png', '_12_real_B.png', '_13_real_B.png')
pass_brains = {}
for subdir, dirs, files in os.walk(dir_pass):
    for file in files:
        if file.split("_")[0] not in pass_brains.keys():
            pass_brains[file.split("_")[0]] = 0
        if file.endswith(images):
            img_real_diff = Image.open(dir_pass + "/" + file)
            img_fake_diff = Image.open(dir_pass + "/" + file.replace("real", "fake"))
            im1 = imagehash.average_hash(img_real_diff, hash_size=256)
            im2 = imagehash.average_hash(img_fake_diff, hash_size=256)
            pass_brains[file.split("_")[0]] = pass_brains[file.split("_")[0]] + (abs(im1 - im2))

print("Diff Pass Means: " + str(numpy.array(list(pass_brains.values())).mean()))

fail_brains = {}
for subdir, dirs, files in os.walk(dir_fail):
    for file in files:
        if file.split("_")[0] not in fail_brains.keys():
            fail_brains[file.split("_")[0]] = 0
        if file.endswith(images):
            img_real_diff = Image.open(dir_fail + "/" + file)
            img_fake_diff = Image.open(dir_fail + "/" + file.replace("real", "fake"))
            im1 = imagehash.average_hash(img_real_diff, hash_size=256)
            im2 = imagehash.average_hash(img_fake_diff, hash_size=256)
            fail_brains[file.split("_")[0]] = fail_brains[file.split("_")[0]] + (abs(im1 - im2))

print("Diff Fail Means: " + str(numpy.array(list(fail_brains.values())).mean()))
pass_accs = []
fail_accs = []
cutoffs = [3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500]
for cutoff in cutoffs:
    print("Cutoff: " + str(cutoff))
    print("pass_brains:")
    print(str(len([k for k, v in pass_brains.items() if float(v) <= cutoff])) + "/" + str(len(pass_brains)))
    pass_accs.append(len([k for k, v in pass_brains.items() if float(v) <= cutoff]) / len(pass_brains))
    print(str(len([k for k, v in pass_brains.items() if float(v) <= cutoff]) / len(pass_brains)))
    print("fail_brains:")
    print(str(len([k for k, v in fail_brains.items() if float(v) >= cutoff])) + "/" + str(len(fail_brains)))
    fail_accs.append(len([k for k, v in fail_brains.items() if float(v) >= cutoff]) / len(fail_brains))
    print(str(len([k for k, v in fail_brains.items() if float(v) >= cutoff]) / len(fail_brains)))
    print("---")

fig = pyplot.figure()
sub = fig.add_subplot(111)
sub.plot(cutoffs, pass_accs, c='b', label='Pass Acc')
sub.plot(cutoffs, fail_accs, c='r', label='Fail Acc')
pyplot.xlabel('Image Difference Value Threshold')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.show()

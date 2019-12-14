import os
import numpy
import imagehash
from PIL import Image
from matplotlib import pyplot

dir_pass = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest_pass/images"
dir_fail = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest/images"
dir_pass_diff = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest_pass/images_diff"
dir_fail_diff = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/results/cervoai_pix2pix_5000_subset/test_latest/images_diff"

# images = ('_7_real_B.png', '_8_real_B.png', '_9_real_B.png', '_10_real_B.png', '_11_real_B.png', '_12_real_B.png', '_13_real_B.png')
images = ('_4_real_B.png', '_5_real_B.png', '_6_real_B.png', '_7_real_B.png', '_8_real_B.png', '_9_real_B.png', '_10_real_B.png',
          '_11_real_B.png', '_12_real_B.png', '_13_real_B.png', '_14_real_B.png', '_15_real_B.png', '_16_real_B.png')
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

fail_test_size = int(len(fail_brains)*0.2)
pass_test_size = int(len(pass_brains)*0.2)
train_fail_acc = []
train_pass_acc = []
test_fail_acc = []
test_pass_acc = []
for i in range(5):
    ft_i = i * fail_test_size
    ft_j = ft_i + fail_test_size
    fail_brains_test = dict(list(fail_brains.items())[ft_i:ft_j])
    fail_brains_train = dict(list(fail_brains.items())[0:ft_i] + list(fail_brains.items())[ft_j:-1])

    pt_i = i * pass_test_size
    pt_j = pt_i + pass_test_size
    pass_brains_test = dict(list(pass_brains.items())[pt_i:pt_j])
    pass_brains_train = dict(list(pass_brains.items())[0:pt_i] + list(pass_brains.items())[pt_j:-1])

    pass_accs = []
    fail_accs = []
    cutoffs = range(3500, 9000, 100)
    # cutoffs = [3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500]

    for cutoff in cutoffs:
        print("Cutoff: " + str(cutoff))
        print("train_pass_brains:")
        print(str(len([k for k, v in pass_brains_train.items() if float(v) <= cutoff])) + "/" + str(len(pass_brains_train)))
        pass_accs.append(len([k for k, v in pass_brains_train.items() if float(v) <= cutoff]) / len(pass_brains_train))
        print(str(len([k for k, v in pass_brains_train.items() if float(v) <= cutoff]) / len(pass_brains_train)))
        print("train_fail_brains:")
        print(str(len([k for k, v in fail_brains_train.items() if float(v) >= cutoff])) + "/" + str(len(fail_brains_train)))
        fail_accs.append(len([k for k, v in fail_brains_train.items() if float(v) >= cutoff]) / len(fail_brains_train))
        print(str(len([k for k, v in fail_brains_train.items() if float(v) >= cutoff]) / len(fail_brains_train)))
        print("---")
    train_pass_acc.append(pass_accs)
    train_fail_acc.append(fail_accs)

    fig = pyplot.figure()
    sub = fig.add_subplot(111)
    sub.plot(cutoffs, pass_accs, c='b', label='Pass Acc')
    sub.plot(cutoffs, fail_accs, c='r', label='Fail Acc')
    pyplot.xlabel('Image Difference Value Threshold')
    pyplot.ylabel('Accuracy')
    pyplot.title('Train ' + str(i + 1))
    pyplot.legend()
    pyplot.show()

    pass_accs = []
    fail_accs = []
    for cutoff in cutoffs:
        print("Cutoff: " + str(cutoff))
        print("test_pass_brains:")
        print(str(len([k for k, v in pass_brains_test.items() if float(v) <= cutoff])) + "/" + str(len(pass_brains_test)))
        pass_accs.append(len([k for k, v in pass_brains_test.items() if float(v) <= cutoff]) / len(pass_brains_test))
        print(str(len([k for k, v in pass_brains_test.items() if float(v) <= cutoff]) / len(pass_brains_test)))
        print("test_fail_brains:")
        print(str(len([k for k, v in fail_brains_test.items() if float(v) >= cutoff])) + "/" + str(len(fail_brains_test)))
        fail_accs.append(len([k for k, v in fail_brains_test.items() if float(v) >= cutoff]) / len(fail_brains_test))
        print(str(len([k for k, v in fail_brains_test.items() if float(v) >= cutoff]) / len(fail_brains_test)))
        print("---")
    test_pass_acc.append(pass_accs)
    test_fail_acc.append(fail_accs)

    fig = pyplot.figure()
    sub = fig.add_subplot(111)
    sub.plot(cutoffs, pass_accs, c='b', label='Pass Acc')
    sub.plot(cutoffs, fail_accs, c='r', label='Fail Acc')
    pyplot.xlabel('Image Difference Value Threshold')
    pyplot.ylabel('Accuracy')
    pyplot.title('Test ' + str(i + 1))
    pyplot.legend()
    pyplot.show()

temp_train_pass_acc = [sum(x) for x in zip(*train_pass_acc)]
train_pass_acc_means = [x / 5 for x in temp_train_pass_acc]

temp_train_fail_acc = [sum(x) for x in zip(*train_fail_acc)]
train_fail_acc_means = [x / 5 for x in temp_train_fail_acc]

temp_test_pass_acc = [sum(x) for x in zip(*test_pass_acc)]
test_pass_acc_means = [x / 5 for x in temp_test_pass_acc]

temp_test_fail_acc = [sum(x) for x in zip(*test_fail_acc)]
test_fail_acc_means = [x / 5 for x in temp_test_fail_acc]

fig = pyplot.figure()
sub = fig.add_subplot(111)
sub.plot(cutoffs, train_pass_acc_means, c='b', label='Pass Acc')
sub.plot(cutoffs, train_fail_acc_means, c='r', label='Fail Acc')
pyplot.xlabel('Image Difference Value Threshold')
pyplot.ylabel('Accuracy')
pyplot.title('Train with kfold k=5')
pyplot.legend()
pyplot.show()

fig = pyplot.figure()
sub = fig.add_subplot(111)
sub.plot(cutoffs, test_pass_acc_means, c='b', label='Pass Acc')
sub.plot(cutoffs, test_fail_acc_means, c='r', label='Fail Acc')
pyplot.xlabel('Image Difference Value Threshold')
pyplot.ylabel('Accuracy')
pyplot.title('Test with kfold k=5')
pyplot.legend()
pyplot.show()


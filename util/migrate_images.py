import os
import pandas as pd
import shutil

src = "C:/Users/jeffp/CervoAI/Data/raw/AI_FS_QC_img"
# dst = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/datasets/cervoai_pix2pix"
dst = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/datasets/cervoai_pix2pix_axial"
csv = "C:/Users/jeffp/CervoAI/Data/raw/AI_FS_QC_img/data_AI_QC.csv"
dstAtrain = dst + "/A/train"
dstBtrain = dst + "/B/train"
dstAtest = dst + "/A/test"
dstBtest = dst + "/B/test"
dstAval = dst + "/A/val"
dstBval = dst + "/B/val"

n_xt_train = 5000
n_xt_test = 1000
n_xt_val = 500
n_xt_total = n_xt_train + n_xt_test + n_xt_val

df = pd.read_csv(csv)
fails_to_not_migrate = df.loc[df['fail'] == 1]
fails_to_not_migrate = fails_to_not_migrate["id"].values

i_t1 = 0
for subdir, dirs, files in os.walk(src):
    if "Axial/t1" in subdir.replace("\\", "/"):
        if i_t1 > n_xt_total:
            break
        if subdir.replace("\\", "/").strip("Axial/t1").strip(src) not in fails_to_not_migrate:
            print("Copying t1: " + str(i_t1) + "/" + str(n_xt_total))
            i_t1 = i_t1 + 1
            for file in files:
                if i_t1 < n_xt_train:
                    shutil.copy(subdir.replace("\\", "/") + "/" + file, dstAtrain)
                elif i_t1 >= n_xt_train and i_t1 <= (n_xt_train + n_xt_test):
                    shutil.copy(subdir.replace("\\", "/") + "/" + file, dstAtest)
                else:
                    shutil.copy(subdir.replace("\\", "/") + "/" + file, dstAval)

i_labels = 0
for subdir, dirs, files in os.walk(src):
    if "Axial/labels" in subdir.replace("\\", "/"):
        if i_labels > n_xt_total:
            break
        if subdir.replace("\\", "/").strip("Axial/labels").strip(src) not in fails_to_not_migrate:
            print("Copying labels: " + str(i_labels) + "/" + str(n_xt_total))
            i_labels = i_labels + 1
            for file in files:
                if i_labels < n_xt_train:
                    shutil.copy(subdir.replace("\\", "/") + "/" + file, dstBtrain)
                elif i_labels >= n_xt_train and i_labels <= (n_xt_train + n_xt_test):
                    shutil.copy(subdir.replace("\\", "/") + "/" + file, dstBtest)
                else:
                    shutil.copy(subdir.replace("\\", "/") + "/" + file, dstBval)

import os
import pandas as pd
import shutil

src = "C:/Users/jeffp/CervoAI/Data/raw/AI_FS_QC_img"
dst = "C:/Users/jeffp/pytorch-CycleGAN-and-pix2pix/datasets/cervoai_pix2pix"
csv = "C:/Users/jeffp/CervoAI/Data/raw/AI_FS_QC_img/data_AI_QC.csv"
dstAtest = dst + "/A/test_fail"
dstBtest = dst + "/B/test_fail"

df = pd.read_csv(csv)
fails_to_not_migrate = df.loc[df['fail'] == 1]
fails_to_not_migrate = fails_to_not_migrate["id"].values

i_t1 = 0
for subdir, dirs, files in os.walk(src):
    if "Coronal/t1" in subdir.replace("\\", "/"):
        if subdir.replace("\\", "/").strip("Coronal/t1").strip(src) in fails_to_not_migrate:
            print("Copying t1: " + str(i_t1) + "/565")
            i_t1 = i_t1 + 1
            for file in files:
                shutil.copy(subdir.replace("\\", "/") + "/" + file, dstAtest)


i_labels = 0
for subdir, dirs, files in os.walk(src):
    if "Coronal/labels" in subdir.replace("\\", "/"):
        if subdir.replace("\\", "/").strip("Coronal/labels").strip(src) in fails_to_not_migrate:
            print("Copying labels: " + str(i_labels) + "/565")
            i_labels = i_labels + 1
            for file in files:
                shutil.copy(subdir.replace("\\", "/") + "/" + file, dstBtest)

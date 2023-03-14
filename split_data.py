import os
from numpy import random
import shutil
# reference: https://cs230.stanford.edu/blog/split/

rgb_dir = './data/2023-02-23-03-16/0/rgb'
label_dir = './data/2023-02-23-03-16/0/label'
rgb_filenames = next(os.walk(rgb_dir))[2]
label_filenames = next(os.walk(label_dir))[2]
rgb_filenames.sort()  # make sure that the filenames have a fixed order before shuffling
label_filenames.sort() 

seed_num = 666
random.seed(seed_num)
random.shuffle(rgb_filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

random.seed(seed_num)
random.shuffle(label_filenames)

# 70-20-10 train-val-test
split_1 = int(0.7 * len(rgb_filenames))
split_2 = int(0.9 * len(rgb_filenames))
train_rgb_filenames = rgb_filenames[:split_1]
val_rgb_filenames = rgb_filenames[split_1:split_2]
test_rgb_filenames = rgb_filenames[split_2:]

train_label_filenames = label_filenames[:split_1]
val_label_filenames = label_filenames[split_1:split_2]
test_label_filenames = label_filenames[split_2:]

def copy_paste_files(filenames, origin_dir, target_dir):
    for file in filenames:
        origin = origin_dir + "/"+ file
        target = target_dir + "/" + file
        shutil.copy(origin, target)

# create directories
if not os.path.exists("./train"):
    os.makedirs("./train/images")
    os.makedirs("./train/labels")

if not os.path.exists("./val"):
    os.makedirs("./val/images")
    os.makedirs("./val/labels")

if not os.path.exists("./test"):
    os.makedirs("./test/images")
    os.makedirs("./test/labels")

# copy paste files
target_dir_split = ["./train", "./val", "./test"]
rgb_filenames_split = [train_rgb_filenames, val_rgb_filenames, test_rgb_filenames]
label_filenames_split = [train_label_filenames, val_label_filenames, test_label_filenames]
for k in range(3):
    copy_paste_files(rgb_filenames_split[k], rgb_dir, target_dir_split[k] + "/images")
    copy_paste_files(label_filenames_split[k], label_dir, target_dir_split[k] + "/labels")
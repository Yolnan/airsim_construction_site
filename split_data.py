import os
from numpy import random
import shutil
# reference: https://cs230.stanford.edu/blog/split/

def copy_paste_rename(filenames, origin_dir, target_dir, class_id):
        for file in filenames:
            origin = origin_dir + "/"+ file
            target = target_dir + "/" + class_id + "_"  + file
            shutil.copy(origin, target)


if __name__ == "__main__":
    parent_folder = "./data/forklift1_img900"
    class_id_list = next(os.walk(parent_folder))[1]
    split = [70, 20]   # train, val percentages; test = 100-train-val

    # create directories
    if not os.path.exists("./train"):
        os.makedirs("./train/images")
        os.makedirs("./train/labels")

    if not os.path.exists("./val"):
        os.makedirs("./val/images")
        os.makedirs("./val/labels")

    if not os.path.exists("./test") and sum(split) < 100:
        os.makedirs("./test/images")
        os.makedirs("./test/labels")

    for class_id in class_id_list:
        rgb_dir = parent_folder + f"/{class_id}/rgb"
        label_dir = parent_folder + f"/{class_id}/label"
        rgb_filenames = next(os.walk(rgb_dir))[2]
        label_filenames = next(os.walk(label_dir))[2]
        rgb_filenames.sort()  # make sure that the filenames have a fixed order before shuffling
        label_filenames.sort() 

        seed_num = 666
        random.seed(seed_num)
        random.shuffle(rgb_filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

        random.seed(seed_num)
        random.shuffle(label_filenames)

        # split filenames
        split_1 = int(split[0]/100 * len(rgb_filenames))
        split_2 = int((split[0] + split[1])/100 * len(rgb_filenames))
        train_rgb_filenames = rgb_filenames[:split_1]
        val_rgb_filenames = rgb_filenames[split_1:split_2]
        test_rgb_filenames = rgb_filenames[split_2:]

        train_label_filenames = label_filenames[:split_1]
        val_label_filenames = label_filenames[split_1:split_2]
        test_label_filenames = label_filenames[split_2:]

        # copy paste files
        target_dir_split = ["./train", "./val", "./test"]
        rgb_filenames_split = [train_rgb_filenames, val_rgb_filenames, test_rgb_filenames]
        label_filenames_split = [train_label_filenames, val_label_filenames, test_label_filenames]
        for k in range(3):
            copy_paste_rename(rgb_filenames_split[k], rgb_dir, target_dir_split[k] + "/images", class_id)
            copy_paste_rename(label_filenames_split[k], label_dir, target_dir_split[k] + "/labels", class_id)
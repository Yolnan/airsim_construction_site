import cv2
import numpy as np
import os
# Add backgrounds to synthetic image with construction vehicle with simple background
# recommend background image resolution to no less than synthetic image resolution

def replace_background(rgb_path, mask_path, back_path, output_folder, preview):
    # read images
    rgb = cv2.imread(rgb_path)
    mask = cv2.imread(mask_path)
    back = cv2.imread(back_path)

     # resize image if necessary
    if back.shape[0] != rgb.shape[0] or back.shape[1] != rgb.shape[1]: 
        back  = cv2.resize(back, dsize=(rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    
    # stitch rgb and back images
    rgb[mask < 1] = back[mask < 1]
    output_path = output_folder + "/" + (os.path.basename(rgb_path))

    # save or preview stitched image
    if preview is True:
        cv2.imshow("stitched image", rgb)
        print("Save path: " + output_path)
        cv2.waitKey(0)
    else:
        cv2.imwrite(output_path, rgb)

if __name__ == "__main__":
    parent_folder = "./data/forklift6_img360"
    background_folder = "./forklift_background"
    class_id_list = next(os.walk(parent_folder))[1]
    back_list = os.listdir(background_folder)
    preview = True

    for class_id in class_id_list:
        rgb_folder = parent_folder + f"/{class_id}/rgb"
        mask_folder = parent_folder + f"/{class_id}/mask"
        output_folder = parent_folder + f"/{class_id}/rgb_newback"
        if preview is False:
            os.makedirs(output_folder)
        for filename in os.listdir(r"./"+rgb_folder):
            rgb_path = rgb_folder + "/" + filename
            mask_path = mask_folder + "/" + filename
            back_path = background_folder + "/" + np.random.choice(back_list)
            replace_background(rgb_path, mask_path, back_path, output_folder, preview)
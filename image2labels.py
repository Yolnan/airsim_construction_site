import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation
from icp import convert_pose_to_euler

def contour_to_percent_points(contour: list, img_width: float, img_height: float):
    percent_points = [[point_wrapper[0][0]/img_height, point_wrapper[0][1]/img_width] for point_wrapper in contour]
   
    return percent_points

def points_format(points: list, extra: str):
    output = [f"{extra}"] + list(map(lambda point: f"{point[0]:.8f} {point[1]:.8f}", points))
    return " ".join(output)
    
def save_contours(path: str, contours: list, img_width: float, img_height: float, extra: str):
    output = []
    for contour in contours:
        output.append(points_format(contour_to_percent_points(contour, img_width, img_height), extra))
    with open(path, "w") as f:
        print(path)
        f.write("\n".join(output))

def pose_to_class_id(euler_angles):
    z_angle = euler_angles[0]
    # "0"->rear, "1"->left, "2"->right, "3"->front
    if z_angle < 25 and z_angle > -25:
        class_id = "0"
    elif z_angle >=25 and z_angle <= 155:
        class_id = "1"
    elif z_angle >= -155 and z_angle <= -25:
        class_id = "2"
    else:
        class_id = "3"
    return class_id


if __name__ == "__main__":
    parent_folder = "./data/forklift1_img900"
    class_id_list = next(os.walk(parent_folder))[1]
    preview_labels = True  # False: save labels, True: preview labels by viewing contours of mask
    classname_key = [str(x) for x in range(4)]
    classname_str = ["rear", "left", "right", "front"]
    classname_dict = {classname_key[i]: classname_str[i] for i in range(len(classname_key))}
    for class_id in class_id_list:
        mask_folder = parent_folder + f"/{class_id}/mask"
        pose_folder = parent_folder + f"/{class_id}/pose"
        label_folder = parent_folder + f"/{class_id}/label"
        if not os.path.exists(label_folder) and preview_labels is False:
            os.makedirs(label_folder)

        for filename in os.listdir(r"./"+mask_folder):
            array_of_img = cv2.imread(mask_folder + "/" + filename) # 3-channel image
            W = array_of_img.shape[0]  #480
            H = array_of_img.shape[1]  #640
            array_of_img1 = cv2.cvtColor(array_of_img, cv2.COLOR_BGR2GRAY)
            ret, bin_image = cv2.threshold(array_of_img1, 127, 255, cv2.THRESH_BINARY)
            
            # dilate and erode
            kernel = np.ones((11, 11), dtype=np.uint8)
            dilate = cv2.dilate(bin_image, kernel, 1)
            erosion = cv2.erode(dilate, kernel, iterations=1)

            mask = np.zeros([W+2, H+2],np.uint8)
            im_floodfill = erosion.copy()
            isbreak = False
            for i in range(im_floodfill.shape[0]):
                for j in range(im_floodfill.shape[1]):
                    if(im_floodfill[i][j]==0):
                        seedPoint=(i,j)
                        isbreak = True
                        break
                if(isbreak):
                    break

            cv2.floodFill(im_floodfill, mask,seedPoint, 255)
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)
            im_out1 = erosion | im_floodfill_inv

            kernel = np.ones((5, 5), dtype=np.uint8)
            erosion = cv2.erode(im_out1, kernel, iterations=1)
            im_out1 = cv2.dilate(erosion, kernel, 1)
            
            # find contours of mask image
            contours, hierarchy = cv2.findContours(im_out1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # determine class_id for labeling (rear, left, right, front view)
            euler_angles = convert_pose_to_euler(pose_folder + "/" + os.path.splitext(filename)[0] + ".txt")
            label_class_id = pose_to_class_id(euler_angles)

            # save or visualize mask labels
            if preview_labels is False:
                save_contours(label_folder + "/" + os.path.splitext(filename)[0] + ".txt", contours, W, H, label_class_id)
            else:
                cv2.drawContours(array_of_img,contours,-1,(0,0,255),3)  
                print("save to: " + label_folder + "/" + os.path.splitext(filename)[0] + ".txt")
                print("view: " + classname_dict[label_class_id])
                cv2.imshow("img", array_of_img)  
                cv2.waitKey(0) 
# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
# This is modified based on cv_mode.py

from random import shuffle
import sys
import airsim

import pprint
import os
import time
from datetime import datetime
import numpy as np
import cv2
import transforms3d as tfm
import pandas as pd
from matplotlib import pyplot as plt

def getArrayFromAirsimImage(rgb, depth):
    img1d = np.frombuffer(rgb.image_data_uint8, dtype=np.uint8)
    img1d_depth = np.array(depth.image_data_float,
                           dtype="float32")
    # if rgb.height == 0:
    #     return -1, -1
    # reshape array to 4 channel i mage array H X W X 4
    img_rgb = img1d.reshape(rgb.height, rgb.width, 3)
    img_rgb_filp = img_rgb.copy()
    img_depth = img1d_depth.reshape(depth.height,
                                    depth.width)
    return img_rgb_filp, img_depth

def airsimPoseToMat(pose):
    pos = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
    ori = tfm.quaternions.quat2mat(
        [pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val])
    p = np.concatenate((ori, np.array([pos]).T), axis=1)
    return np.concatenate((p, np.array([[0, 0, 0, 1]])), axis=0)

def extract_save_image(cnt, save_folder_path, object_index):
    infos = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True),
                                airsim.ImageRequest(
                                    camera_name, airsim.ImageType.Segmentation, False, False),
                                airsim.ImageRequest(
                                    camera_name, airsim.ImageType.Scene, False, False)
                                 ])
    depth = infos[0]
    seg = infos[1]
    rgb = infos[2]
    img_rgb, img_depth = getArrayFromAirsimImage(rgb, depth)
    cv2.imwrite(
        f"{save_folder_path}{file_deli}rgb{file_deli}{cnt}.png", img_rgb)
    depth_filter_range = 60  # meter
    img_depth = np.where(img_depth > depth_filter_range, 0., img_depth)
    img_depth *= 1000  # save depth image using millimeter
    cv2.imwrite(
        f"{save_folder_path}{file_deli}depth{file_deli}{cnt}.png", img_depth.astype(np.uint16))

    img_seg, img_depth = getArrayFromAirsimImage(seg, depth)
    # plt.imshow(cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB))
    # plt.show()
    mask = np.ma.getmaskarray(np.ma.masked_equal(img_seg, seg_bgr[object_index]))
    mask_img = mask[:, :, 0] * img_seg[:, :, 0]
    mask_img = np.where(mask_img > 0, 255, mask_img)
    cv2.imwrite(
        f"{save_folder_path}{file_deli}mask{file_deli}{cnt}.png", mask_img)

def generate_object_data(object_index):
    print(
        f"Start to generate data for object: {unreal_id_ls[object_index]}")
    # create folder
    object_name = unreal_id_ls[object_index]
    save_folder_path = root_folder + f'{file_deli}{object_index}'
    os.mkdir(save_folder_path)
    os.mkdir(save_folder_path + file_deli + 'depth')
    os.mkdir(save_folder_path + file_deli + 'rgb')
    os.mkdir(save_folder_path + file_deli + 'pose')
    os.mkdir(save_folder_path + file_deli + 'mask')

    initial_pose_set = False
    cnt = 0
    exit_flag = False
    shuffle_flag = True

    camera_num = 5 # This represents different camera heights
    vehicle_orientations = 36
    closeness = 5
    index_list = np.array([x for x in range(camera_num * vehicle_orientations * closeness)])
    if shuffle_flag:
        np.random.shuffle(index_list)
    base2center = np.array(
        [[1., 0., 0., 18.], 
        [0., 1., 0., 25.], 
        [0., 0., 1., 0.5], 
        [0., 0., 0., 1.]])

    # random cam
    for j in range(camera_num):
        if j>0:
            break
        if exit_flag:
            break
        center_cam_ori_x = 0 
        center_cam_ori_y =  -0.01 #np.random.uniform() * np.pi * (-0.3) - 0.1 #-1 *(np.pi -0.5) #
        center_cam_ori_z = 0 #np.random.uniform() * np.pi * 2

        center_cam_ori = tfm.euler.euler2mat(
            center_cam_ori_z, center_cam_ori_y, center_cam_ori_x, "szyx") #in NED coordinates

        for k in range(9, 9 + closeness):
            if k > 9:
                break
            dis =  k #np.random.uniform() * 5 + 8
            # make the x axis of the camera pointing to the center

            center_cam_pos = -dis * np.expand_dims(center_cam_ori[:, 0], 1)
            center_cam_pos[-1] -= j

            center2cam = np.concatenate(
                [np.concatenate([center_cam_ori, center_cam_pos], 1), [[0, 0, 0, 1]]], 0)
            

            base2cam = base2center @ center2cam
            camera_quat = tfm.quaternions.mat2quat(
                base2cam[:3, :3]).tolist()  # [w, x, y, z]
            camera_posi = base2cam[:3, 3].tolist()
            # airsim.Quaternionr accepts [x, y, z, w]
            airsim_camera_pose = airsim.Pose(airsim.Vector3r(camera_posi[0], camera_posi[1], camera_posi[2]),
                                            airsim.Quaternionr(camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]))
            client.simSetCameraPose(camera_name, airsim_camera_pose)
            print(f"Setting camera pose {j}: {base2cam}")
            time.sleep(3)
            actual_cam_pose = airsimPoseToMat(
                client.simGetCameraInfo(camera_name).pose)
            print(f"Actual camera pose {j}: {actual_cam_pose}")
            # Need to re-calculate the center position because of the error in camera position
            # The translation below only applys to specific object settings and randomization strategy
            sin_pitch = actual_cam_pose[2][0]
            cos_pitch = np.sqrt(1 - sin_pitch * sin_pitch)
            dz = base2center[2][3] - actual_cam_pose[2][3]
            dx = dz / sin_pitch * cos_pitch / cos_pitch * actual_cam_pose[0][0]
            dy = dz / sin_pitch * cos_pitch / cos_pitch * actual_cam_pose[1][0]
            actual_center_pos = [actual_cam_pose[0][3] + dx,
                                actual_cam_pose[1][3] + dy, actual_cam_pose[2][3] + dz]
            print("actual_cen: \n",actual_center_pos )

            for i in range(vehicle_orientations):
                if i >0:
                    break
                x_displace = 0 # np.random.uniform() * 4
                y_displace = 0 # np.random.uniform() * 4
                rz = (i+1) *10 * np.pi /180. #np.random.uniform() * np.pi * 2
                # [w, x, y, z]
                center_vehicle_ori = tfm.euler.euler2quat(
                    rz, 0, 0, "rzyx").tolist()
                if not initial_pose_set: 
                    vehicle_posi = actual_center_pos#why is the vehicle pos set to actual center pos
                    initial_pose_set = not initial_pose_set
                airsim_object_pose = airsim.Pose(airsim.Vector3r(vehicle_posi[0] + x_displace, vehicle_posi[1] + y_displace, vehicle_posi[2]),
                                                airsim.Quaternionr(center_vehicle_ori[1], center_vehicle_ori[2], center_vehicle_ori[3], center_vehicle_ori[0]))
                if not client.simSetObjectPose(object_name, airsim_object_pose):
                    print(
                        "Failed to set vehicle pose, check if you set the object mobility to movable! Quit...")
                    exit_flag = True
                    break
                time.sleep(1)

                actual_obj_pose = airsimPoseToMat(
                    client.simGetObjectPose(object_name))
                actual_cam_pose = airsimPoseToMat(
                    client.simGetCameraInfo(camera_name).pose)

                extract_save_image(index_list[cnt], save_folder_path, object_index)
                cam_wrt_obj = np.linalg.inv(actual_obj_pose) @ actual_cam_pose
                np.savetxt(
                    f"{save_folder_path}{file_deli}pose{file_deli}{index_list[cnt]}.txt", cam_wrt_obj)
                print(f"Save #{cnt} data with index {index_list[cnt]}\n")
                cnt += 1
    origin_object_pose = airsim.Pose(airsim.Vector3r(-20, 10 * object_index, 0),
                                     airsim.Quaternionr(0, 0, 0, 1))
    client.simSetObjectPose(object_name, origin_object_pose)


if __name__ == "__main__":
    # airsim setup
    pp = pprint.PrettyPrinter(indent=4)
    client = airsim.VehicleClient()
    client.confirmConnection()
    camera_name = "0"
    unreal_id_ls = ["SM_Forklift_2", 
                    "SM_MERGED_GC_CAT_DP70_192", 
                    "SM_MERGED_Toyota_forklift_11",
                    "SM_MERGED_Forklift-Mitsubishi-FGC30N_8",
                    "SM_MERGED_EmpilhadeiraMod_2",
                    "SM_MERGED_Forklift_Sporty_5"] # Unreal ID Name, not label
    
    # create convert seg_rgbs.txt to list, from: https://microsoft.github.io/AirSim/seg_rgbs.txt
    # df = pd.read_csv("./seg_rgbs.txt", delimiter="\t", header=None)
    # seg_bgr = []
    # for ind in range(len(df.index)):
    #     rgb_string = df.iloc[ind, 1]
    #     rgb_string = rgb_string.replace("[", "")
    #     rgb_string = rgb_string.replace("]", "")
    #     rgb_label = np.fromstring(rgb_string, dtype=np.uint8, sep=",")
    #     seg_bgr.append(np.flip(rgb_label))
    seg_bgr = [np.array([250, 20, 234]), 
               np.array([109, 153, 222]), 
               np.array([30, 161, 114]),
               np.array([84, 95, 31]),
               np.array([249, 186, 224]),
               np.array([181, 109, 101])]   # hardcoded segmentation color, setting colorID seems to be broken in AirSim, always returns -1

    # create root folder
    file_deli = '\\'
    time_str = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M")
    root_folder = os.path.abspath("data") + file_deli + time_str
    os.mkdir(root_folder)

    airsim.wait_key('Press any key to start loop')

    print(f"First put all the vehicles away from center")
    for obj_ind in range(len(unreal_id_ls)):
        object_name = unreal_id_ls[obj_ind]
        airsim_object_pose = airsim.Pose(airsim.Vector3r(-20, 10 * obj_ind, 0),
                                        airsim.Quaternionr(0, 0, 0, 1))
        if not client.simSetObjectPose(object_name, airsim_object_pose):
            print(
                f"Failed to set vehicle pose, check if you set the object {object_name} mobility to movable! Quit...")
            exit(1)
        time.sleep(1)

    for obj_ind in range(len(unreal_id_ls)):
        generate_object_data(obj_ind)
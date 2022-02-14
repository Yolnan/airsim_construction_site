# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
# This is modified based on cv_mode.py

from operator import index
from random import shuffle
import sys
import airsim

import pprint
import os
import time
from datetime import datetime
import math
import tempfile
from IPython import embed
import numpy as np
import cv2
import keyboard
import transforms3d as tfm


def getArrayFromAirsimImage(rgb, depth):
    img1d = np.frombuffer(rgb.image_data_uint8, dtype=np.uint8)
    img1d_depth = np.array(depth.image_data_float,
                           dtype="float32")
    # if rgb.height == 0:
    #     return -1, -1
    # reshape array to 4 channel image array H X W X 4
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


pp = pprint.PrettyPrinter(indent=4)

client = airsim.VehicleClient()
client.confirmConnection()

camera_name = "0"
object_index = 1
object_name = "SM_ForkLift5"
file_deli = '\\'

# client.simGetObjectPose("SM_ForkLift5")
# client.simGetCameraInfo("0")

time_str = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M")
save_folder_path = os.path.abspath("data") + file_deli + time_str
os.mkdir(save_folder_path)
save_folder_path += f'{object_index}'
os.mkdir(save_folder_path)
os.mkdir(save_folder_path + file_deli + 'depth')
os.mkdir(save_folder_path + file_deli + 'rgb')
os.mkdir(save_folder_path + file_deli + 'pose')
os.mkdir(save_folder_path + file_deli + 'mask')


def extract_save_image(cnt):
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
    cv2.imwrite(
        f"{save_folder_path}{file_deli}depth{file_deli}{cnt}.png", img_depth.astype(np.uint16))

    img_seg, img_depth = getArrayFromAirsimImage(seg, depth)
    mask = np.ma.getmaskarray(np.ma.masked_equal(
        img_seg, np.array([242, 162, 90])))
    mask_img = mask[:, :, 0] * img_seg[:, :, 0]
    mask_img = np.where(mask_img == 242, 255, mask_img)
    cv2.imwrite(
        f"{save_folder_path}{file_deli}mask{file_deli}{cnt}.png", mask_img)


airsim.wait_key(
    'Press any key to start loop')

cnt = 0
exit_flag = False
shuffle_flag = True

# vehicle_pos = np.array([9., 8., -1.])
# # vehicle_ori = [0, 0, 0, 1]
# vehicle_angle = np.array([0., 0., 0.])
# camera_pos = np.array([3., 3., -10.])
# # camera_ori = [0, -0.3826, 0, 0.9238]
# camera_angle = np.array([0., -np.pi / 4, 0.])

camera_num = 2
vehicle_num = 10
index_list = np.array([x for x in range(camera_num * vehicle_num)])
if shuffle_flag:
    np.random.shuffle(index_list)
base2center = np.array(
    [[1., 0., 0., 10.], [0., 1., 0., 10.], [0., 0., 1., -1.], [0., 0., 0., 1.]])

embed()

# random cam
for i in range(camera_num):
    if exit_flag:
        break
    center_cam_ori_x = 0
    center_cam_ori_y = np.random.uniform() * np.pi * -0.4
    center_cam_ori_z = np.random.uniform() * np.pi * 2

    center_cam_ori = tfm.euler.euler2mat(
        center_cam_ori_z, center_cam_ori_y, center_cam_ori_x, "rzyx")
    dis = np.random.uniform() * 5 + 5
    # make the x axis of the camera pointing to the center
    center_cam_pos = -dis * np.expand_dims(center_cam_ori[:, 0], 1)
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
    print(f"Setting camera pose {i}: {base2cam}")
    time.sleep(5)

    for j in range(vehicle_num):
        x_displace = np.random.uniform() * 0
        y_displace = np.random.uniform() * 0
        rz = np.random.uniform() * np.pi * 2
        # [w, x, y, z]
        center_vehicle_ori = tfm.euler.euler2quat(rz, 0, 0, "rzyx").tolist()
        vehicle_posi = base2center[:3, 3].tolist()
        airsim_object_pose = airsim.Pose(airsim.Vector3r(vehicle_posi[0] + x_displace, vehicle_posi[1] + y_displace, vehicle_posi[2]),
                                         airsim.Quaternionr(center_vehicle_ori[1], center_vehicle_ori[2], center_vehicle_ori[3], center_vehicle_ori[0]))
        if not client.simSetObjectPose(object_name, airsim_object_pose):
            print(
                "Failed to set vehicle pose, check if you set the object mobility to movable! Quit...")
            exit_flag = True
            break
        time.sleep(1)

        actual_obj_pose = airsimPoseToMat(client.simGetObjectPose(object_name))
        actual_cam_pose = airsimPoseToMat(
            client.simGetCameraInfo(camera_name).pose)

        extract_save_image(index_list[cnt])
        cam_wrt_obj = np.linalg.inv(actual_obj_pose) @ actual_cam_pose
        np.savetxt(
            f"{save_folder_path}{file_deli}pose{file_deli}{index_list[cnt]}.txt", cam_wrt_obj)
        print(f"Save #{cnt} data with index {index_list[cnt]}\n")
        cnt += 1

        # print("\n\nPress Y to save current scene, N to disgard, Q to finish:\n")

        # while True:
        #     k = keyboard.read_key()
        #     if k == 'N' or k == 'n':
        #         print(f"Discard #{cnt} data\n")
        #         break
        #     elif k == 'Y' or k == 'y':
        #         extract_save_image(cnt)
        #         cam_wrt_obj = np.linalg.inv(actual_obj_pose) @ actual_cam_pose
        #         np.savetxt(
        #             f"{save_folder_path}{file_deli}pose{file_deli}{cnt}.txt", cam_wrt_obj)
        #         print(f"Save #{cnt} data\n")
        #         cnt += 1
        #         break
        #     elif k == 'Q' or k == 'q':
        #         exit_flag = False
        #         break


embed()

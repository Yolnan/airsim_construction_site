# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
# This is modified based on cv_mode.py

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
object_name = "SM_ForkLift5"
file_deli = '\\'

# client.simGetObjectPose("SM_ForkLift5")
# client.simGetCameraInfo("0")

time_str = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M")
save_folder_path = os.path.abspath("data") + file_deli + time_str
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
    'Press any key to set camera-0 gimbal to -15-degree pitch and start loop')
camera_pose = airsim.Pose(airsim.Vector3r(
    0, 0, -9), airsim.to_quaternion(math.radians(-15), 0, 0))  # radians
client.simSetCameraPose(camera_name, camera_pose)

time.sleep(1)
embed()

cnt = 0
exit_flag = True

vehicle_pos = np.array([9., 8., -1.])
# vehicle_ori = [0, 0, 0, 1]
vehicle_angle = np.array([0., 0., 0.])
camera_pos = np.array([3., 3., -10.])
# camera_ori = [0, -0.3826, 0, 0.9238]
camera_angle = np.array([0., -np.pi / 4, 0.])
while exit_flag:
    # TODO: generate random pose here
    vehicle_angle += np.array([0., 0., np.pi / 10])

    # [w x y z]
    vehicle_ori = tfm.euler.euler2quat(*vehicle_angle, 'rxyz')
    camera_ori = tfm.euler.euler2quat(*camera_angle, 'rxyz')

    # airsim does NOT accept numpy.float64
    vehicle_pos_ls = vehicle_pos.tolist()
    vehicle_ori_ls = vehicle_ori.tolist()
    camera_pos_ls = camera_pos.tolist()
    camera_ori_ls = camera_ori.tolist()

    print(vehicle_ori_ls)

    object_pose = airsim.Pose(airsim.Vector3r(vehicle_pos_ls[0], vehicle_pos_ls[1], vehicle_pos_ls[2]), airsim.Quaternionr(
        vehicle_ori_ls[1], vehicle_ori_ls[2], vehicle_ori_ls[3], vehicle_ori_ls[0]))
    if not client.simSetObjectPose(object_name, object_pose):
        print("Failed to set vehicle pose, check if you set the object mobility to movable! Quit...")
        break

    camera_pose = airsim.Pose(airsim.Vector3r(camera_pos_ls[0], camera_pos_ls[1], camera_pos_ls[2]), airsim.Quaternionr(
        camera_ori_ls[1], camera_ori_ls[2], camera_ori_ls[3], camera_ori_ls[0]))
    client.simSetCameraPose(camera_name, camera_pose)
    time.sleep(1)

    actual_obj_pose = airsimPoseToMat(client.simGetObjectPose(object_name))
    actual_cam_pose = airsimPoseToMat(
        client.simGetCameraInfo(camera_name).pose)

    print("\n\nPress Y to save current scene, N to disgard, Q to finish:\n")

    while True:
        k = keyboard.read_key()
        if k == 'N' or k == 'n':
            print(f"Discard #{cnt} data\n")
            break
        elif k == 'Y' or k == 'y':
            extract_save_image(cnt)
            cam_wrt_obj = np.linalg.inv(actual_obj_pose) @ actual_cam_pose
            np.savetxt(
                f"{save_folder_path}{file_deli}pose{file_deli}{cnt}.txt", cam_wrt_obj)
            print(f"Save #{cnt} data\n")
            cnt += 1
            break
        elif k == 'Q' or k == 'q':
            exit_flag = False
            break


embed()

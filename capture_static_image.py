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


pp = pprint.PrettyPrinter(indent=4)

client = airsim.VehicleClient()
client.confirmConnection()

# client.simGetObjectPose("SM_ForkLift5")
# client.simGetCameraInfo("0")

time_str = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M")
save_folder_path = "data/" + time_str
os.mkdir(os.path.abspath(save_folder_path))
os.mkdir(os.path.abspath(save_folder_path + '/depth'))
os.mkdir(os.path.abspath(save_folder_path + '/rgb'))
os.mkdir(os.path.abspath(save_folder_path + '/pose'))
os.mkdir(os.path.abspath(save_folder_path + '/mask'))

def extract_save_image(cnt):
    infos = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True), 
                                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
                                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                                ])
    depth = infos[0]
    seg = infos[1]
    rgb = infos[2]

    img_rgb, img_depth = getArrayFromAirsimImage(rgb, depth) 
    cv2.imwrite(f"{save_folder_path}/rgb/{cnt}.png", img_rgb)
    cv2.imwrite(f"{save_folder_path}/depth/{cnt}.png", img_depth)

    img_seg, img_depth = getArrayFromAirsimImage(seg, depth)
    mask = np.ma.getmaskarray(np.ma.masked_equal(img_seg, np.array([242, 162, 90])))
    mask_img = mask[:, :, 0] * img_seg[:, :, 0]
    mask_img = np.where(mask_img == 242, 255, mask_img)
    cv2.imwrite(f"{save_folder_path}/mask/{cnt}.png", mask_img)

airsim.wait_key('Press any key to set camera-0 gimbal to -15-degree pitch and start loop')
camera_pose = airsim.Pose(airsim.Vector3r(0, 0, -9), airsim.to_quaternion(math.radians(-15), 0, 0)) #radians
client.simSetCameraPose("0", camera_pose)

cnt = 0
exit_flag = True

while exit_flag:
    # TODO: generate random pose here
    vehicle_pos = [9, 8, -1]
    vehicle_ori = [0, 0, 0, 1]
    camera_pos = [3, 3, -10]
    camera_ori = [0, -0.3826, 0, 0.9238]

    object_pose = airsim.Pose(airsim.Vector3r(*vehicle_pos), airsim.Quaternionr(*vehicle_ori))
    client.simSetObjectPose("SM_ForkLift5", object_pose)


    camera_pose = airsim.Pose(airsim.Vector3r(*camera_pos), airsim.Quaternionr(*camera_ori))
    client.simSetCameraPose("0", camera_pose)

    while True:
        k = keyboard.read_key()
        if k == 'N' or k == 'n':
            break
        elif k == 'Y' or k == 'y':
            extract_save_image(cnt)
            # TODO: add save pose txt code
            break
        elif k == 'Q' or k == 'q':
            exit_flag = False
            break

    cnt += 1




embed()



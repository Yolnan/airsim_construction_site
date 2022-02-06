# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
# This is modified based on cv_mode.py

import sys
sys.path.append("../AirSim/PythonClient/computer_vision")
# from pynput import Key, Listener, KeyCode
import setup_path
import airsim

import pprint
import os
import time
import math
import tempfile
from IPython import embed
import numpy as np
import cv2

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

airsim.wait_key('Press any key to set camera-0 gimbal to -15-degree pitch')
camera_pose = airsim.Pose(airsim.Vector3r(0, 0, -9), airsim.to_quaternion(math.radians(-15), 0, 0)) #radians
client.simSetCameraPose("0", camera_pose)

airsim.wait_key('Press any key to get camera parameters')
for camera_name in range(5):
    camera_info = client.simGetCameraInfo(str(camera_name))
    print("CameraInfo %d:" % camera_name)
    pp.pprint(camera_info)

object_pose = airsim.Pose(airsim.Vector3r(9, 8, -1), airsim.Quaternionr(0, 0, 0, 1))
client.simSetObjectPose("SM_ForkLift5", object_pose)
client.simGetObjectPose("SM_ForkLift5")

camera_pose = airsim.Pose(airsim.Vector3r(3, 3, -10), airsim.Quaternionr(0, -0.3826, 0, 0.9238))
client.simSetCameraPose("0", camera_pose)
client.simGetCameraInfo("0")

infos = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True), 
                             airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
                             airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                            ])
depth = infos[0]
seg = infos[1]
rgb = infos[2]

img_rgb, img_depth = getArrayFromAirsimImage(rgb, depth) 
# cv2.imwrite("rgb.png", img_rgb)

img_seg, img_depth = getArrayFromAirsimImage(seg, depth)
mask = np.ma.getmaskarray(np.ma.masked_equal(img_seg, np.array([242, 162, 90])))
mask_img = mask[:, :, 0] * img_seg[:, :, 0]
mask_img = np.where(mask_img == 242, 255, mask_img)
# cv2.imwrite("mask.png", mask_img) 

embed()



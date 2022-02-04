# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
# This is modified based on cv_mode.py

import sys
sys.path.append("./AirSim/PythonClient/computer_vision")
from pynput import keyboard

import setup_path
import airsim

import pprint
import os
import time
import math
import tempfile

pp = pprint.PrettyPrinter(indent=4)

client = airsim.VehicleClient()
client.confirmConnection()

airsim.wait_key('Press any key to set camera-0 gimbal to -15-degree pitch')
camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(-15), 0, 0)) #radians
client.simSetCameraPose("0", camera_pose)

airsim.wait_key('Press any key to get camera parameters')
for camera_name in range(5):
    camera_info = client.simGetCameraInfo(str(camera_name))
    print("CameraInfo %d:" % camera_name)
    pp.pprint(camera_info)
tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_cv_mode")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

airsim.wait_key('Press any key to get images')
all_responses = []
while client.simIsPause():
    # z = x * -20 - 5 # some random number
    # client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(z, z, z), airsim.to_quaternion(x / 3.0, 0, x / 3.0)), True)

    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthVis),
        airsim.ImageRequest("0", airsim.ImageType.Segmentation),
        airsim.ImageRequest("0", airsim.ImageType.Scene)])
        # airsim.ImageRequest("4", airsim.ImageType.DisparityNormalized),
        # airsim.ImageRequest("4", airsim.ImageType.SurfaceNormals)])
    all_responses.append(responses)

print(">>>>>>>>>>>>>>>>>>>>>>>The recording has been stopped<<<<<<<<<<<<<<<<<<<<<<<<<")
for i, responses in enumerate(all_responses):
    for j,response in enumerate(responses):
        filename = os.path.join(tmp_dir, "frame_" + str(i), j)
        if response.pixels_as_float:
            # print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
            airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
        else:
            # print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)

        pose = client.simGetVehiclePose()
        pp.pprint(pose)

        # time.sleep(3)
print(">>>>>>>>>>>>>>>>>>>>>>>The images are saved now<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# currently reset() doesn't work in CV mode. Below is the workaround
# client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)

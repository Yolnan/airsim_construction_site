# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
# This is modified based on cv_mode.py

import sys
sys.path.append("./AirSim/PythonClient/computer_vision")
# from pynput import Key, Listener, KeyCode
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
camera_pose = airsim.Pose(airsim.Vector3r(0, 0, -9), airsim.to_quaternion(math.radians(-15), 0, 0)) #radians
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

video_time = int(input("How many seconds do you want to record?"))

airsim.wait_key('Press any key to get images')
time.sleep(2)
start = time.time()
print("start time: ",start, "s")
end = start
all_responses = []

while end-start < video_time:
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar),
        airsim.ImageRequest("0", airsim.ImageType.Segmentation),
        airsim.ImageRequest("0", airsim.ImageType.Scene)])
    all_responses.append(responses)
    end = time.time()
    print("a frame has been captured, time: ", end, "s")

print(">>>>>>>>>>>>>>>>>>>>>>>The recording has been stopped<<<<<<<<<<<<<<<<<<<<<<<<<")
if not os.path.isdir(os.path.join(tmp_dir,"type0")):
    os.mkdir(os.path.join(tmp_dir,"type0"))
if not os.path.isdir(os.path.join(tmp_dir,"type1")):
    os.mkdir(os.path.join(tmp_dir,"type1"))
if not os.path.isdir(os.path.join(tmp_dir,"type2")):
    os.mkdir(os.path.join(tmp_dir,"type2"))
for i, responses in enumerate(all_responses):
    filename = [os.path.join(tmp_dir,"type0", "frame_" + str(i) +"_type_0"),os.path.join(tmp_dir,"type1", "frame_" + str(i) +"_type_1"),os.path.join(tmp_dir,"type2", "frame_" + str(i) +"_type_2")]
    for j,response in enumerate(responses):
        
        if response.pixels_as_float:
            # print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
            airsim.write_pfm(os.path.normpath(filename[j] + '.pfm'), airsim.get_pfm_array(response))
        else:
            # print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            airsim.write_file(os.path.normpath(filename[j] + '.png'), response.image_data_uint8)

        pose = client.simGetVehiclePose()
        # pp.pprint(pose)

        # time.sleep(3)
print(">>>>>>>>>>>>>>>>>>>>>>>The images are saved now<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# currently reset() doesn't work in CV mode. Below is the workaround
# client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)

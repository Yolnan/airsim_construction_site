
import pandas as pd
from scipy.spatial.transform import Rotation
import cv2
import copy
import open3d as o3d
import numpy as np

# ref: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4459,
    #                                   front=[0.9288, -0.2951, -0.2242],
    #                                   lookat=[1.6784, 2.0612, 1.4451],
    #                                   up=[-0.3402, -0.9189, -0.1996])
    o3d.visualization.draw_geometries([source_temp, target_temp])

def convert_pose_to_euler(pose_path):
    df = pd.read_csv(pose_path, sep=" ")
    R = Rotation.from_matrix(df.iloc[0:3,0:3].to_numpy())
    euler = R.as_euler('ZYX', degrees=True) # unreal seems to use +Z up, +Y forward, +X left (left handed coordinate system)
    return euler

def depth_to_pointcloud(depth, mask, K):
    # extract camera intrinsics parameters
    fx = K[0,0]
    fy = K[1,1]
    ppx = K[0,2]
    ppy = K[1,2]

    # calculate camera matrix inverse
    P_inv = np.eye(4)
    P_inv[0:2,0:3] = np.array([[1/fx, 0, -ppx*fy/(fx*fy)],
                                 [0, 1/fy, -ppy/fy]]) # assume skew is zero
    
    pcd = o3d.geometry.PointCloud()
    points = []

    for u in range(mask.shape[0]):
        for v in range(mask.shape[1]):
            z = depth[u,v]   # get depth info 
            if mask[u,v] > 0:
                
                x = (P_inv[0,0]*v + P_inv[0,2])*z 
                y = (P_inv[1,1]*u + P_inv[1,2])*z 
                # add xyz color point to list
                pt = [z, -x, -y]
                points.append(pt)   
    xyz = np.reshape(points, (len(points), 3))
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

# fake camera matrix, ref: https://github.com/unrealcv/unrealcv/issues/14#issuecomment-487346581
# image_width = 640
# image_height = 480
image_width = 256   # might have used data with wrong resolution
image_height = 144 
camera_fov = 90
f = image_width /(2 * np.tan(camera_fov * np.pi / 360))

Cu = image_width/2
Cv = image_height/2
K = np.array([[f, 0, Cu],
[0, f, Cv],
[0, 0, 1 ]])

pose_path = "./data/2023-02-23-03-16/0/pose/161.txt"
pose = convert_pose_to_euler(pose_path)

depth_path = "./data/2023-02-23-03-16/0/depth/161.png"
depth = cv2.imread(depth_path)

mask_path = "./data/2023-02-23-03-16/0/mask/161.png"
mask = cv2.imread(mask_path)

source_pcd_path = "./SM_Forklift.pcd"
source = o3d.io.read_point_cloud(source_pcd_path)
target = depth_to_pointcloud(depth[:,:,0], mask[:,:,0], K) # airsim mask is 3 channel
trans_init = np.eye(4)
threshold = 0.02

reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
o3d.visualization.draw_geometries([source, target])
# o3d.visualization.draw_geometries([target])
# draw_registration_result(source, target, reg_p2p.transformation)
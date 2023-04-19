
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
    euler_angles = R.as_euler('zyx', degrees=True) # unreal seems to use +Z up, +Y forward, +X left (left handed coordinate system)
    return euler_angles

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
    
    pointcloud = o3d.geometry.PointCloud()
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
    pointcloud.points = o3d.utility.Vector3dVector(xyz)
    return pointcloud


if __name__ == "__main__":
    # fake camera matrix, ref: https://github.com/unrealcv/unrealcv/issues/14#issuecomment-487346581
    image_width = 640   
    image_height = 480 
    camera_fov = 90
    f = image_width /(2 * np.tan(camera_fov * np.pi / 360))

    Cu = image_width/2
    Cv = image_height/2
    K = np.array([[f, 0, Cu],
    [0, f, Cv],
    [0, 0, 1 ]])

    parent_folder = "./data/forklift1_img900/0"
    source_img_num = 0
    source_pose = convert_pose_to_euler(parent_folder + "/" + f"pose/{source_img_num}.txt")
    source_color = cv2.imread(parent_folder + "/" + f"rgb/{source_img_num}.png")
    source_depth = cv2.imread(parent_folder + "/" + f"depth/{source_img_num}.png")
    source_mask = cv2.imread(parent_folder + "/" + f"mask/{source_img_num}.png")
    source = depth_to_pointcloud(source_depth[:,:,0], source_mask[:,:,0], K) # airsim mask is 3 channel

    target_img_num = 49
    target_pose = convert_pose_to_euler(parent_folder + "/" + f"pose/{target_img_num}.txt")
    target_color = cv2.imread(parent_folder + "/" + f"rgb/{target_img_num}.png")
    target_depth = cv2.imread(parent_folder + "/" + f"depth/{target_img_num}.png")
    target_mask = cv2.imread(parent_folder + "/" + f"mask/{target_img_num}.png")
    target = depth_to_pointcloud(target_depth[:,:,0], target_mask[:,:,0], K) # airsim mask is 3 channel

    template_path = "./SM_Forklift.pcd"
    template_pc = o3d.io.read_point_cloud(template_path)

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
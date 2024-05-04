# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:30:56 2024

@author: kevin
"""


import os
from tqdm import tqdm
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import copy

def numerical_sort_key(filename):
    return int(''.join(filter(str.isdigit, filename)))
#
# def rotation_matrix(axis, theta):
#     """
#     Compute the rotation matrix for a given axis and angle (in radians).
#     Axis should be 'x', 'y', or 'z'.
#     """
#     if axis == 'x':
#         return np.array([[1, 0, 0],
#                          [0, np.cos(theta), -np.sin(theta)],
#                          [0, np.sin(theta), np.cos(theta)]])
#     elif axis == 'y':
#         return np.array([[np.cos(theta), 0, np.sin(theta)],
#                          [0, 1, 0],
#                          [-np.sin(theta), 0, np.cos(theta)]])
#     elif axis == 'z':
#         return np.array([[np.cos(theta), -np.sin(theta), 0],
#                          [np.sin(theta), np.cos(theta), 0],
#                          [0, 0, 1]])
#     else:
#         raise ValueError("Axis must be 'x', 'y', or 'z'")
#
# def apply_rotation(pcd, R):
#     """
#     Apply the given rotation matrix R to the point cloud.
#     """
#     # Convert Open3D point cloud to numpy array
#     points = np.asarray(pcd.points)
#     # Apply the rotation
#     rotated_points = np.dot(points, R.T) # Note the transpose of R
#     # Create a new point cloud object for the rotated points
#     rotated_pcd = o3d.geometry.PointCloud()
#     rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)
#     return rotated_pcd
#


def preprocess_point_cloud_2(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 150
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    radius_feature = voxel_size * 200
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size, voxel_multiple, iterations):
    distance_threshold = voxel_size * voxel_multiple
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # RANSAC n points
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(iterations, confidence=1))
    return result


# =============================================================================
# def preprocess_point_cloud(pcd, voxel_size):
#     # Downsample the point cloud
#     pcd_down = pcd.voxel_down_sample(voxel_size)
# 
#     # Estimate normals
#     radius_normal = voxel_size * 2
#     pcd_down.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
# 
#     return pcd_down
# =============================================================================

def pairwise_registration(source, target, result_matrix):
    max_correspondence_distance_coarse = 100
    max_correspondence_distance_fine = 15
    print("Apply point-to-plane ICP")
    try:
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, result_matrix.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    except:
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, result_matrix,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000,
                                                              relative_rmse=1e-6,
                                                              relative_fitness=1e-6))
        
        
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, result_matrix):
    print(len(pcds))
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], result_matrix)
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                print("Odometry")
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))
    return pose_graph


def transform_points(points, R, t):
    # Apply the transformation to the points
    t = np.reshape(t, (1, 3))
    return (R @ points.T).T + t

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation.transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


# Placeholder for the image paths
folder = 'Treasure_Chest/'
images = sorted([img for img in os.listdir(folder) if img.endswith(".png")], key=numerical_sort_key)
image_paths = images[:52]
pc_list = [0, 16, 37, 42]
list_of_point_clouds = []
list_of_poses = []

# Camera parameters (Example values, should be obtained from calibration)
focal_length_mm = 50
sensor_width_mm = 36
image_width_px = 1920
image_height_px = 1080
f_x = (focal_length_mm * image_width_px) / sensor_width_mm
f_y = (focal_length_mm * image_height_px) / sensor_width_mm
c_x = image_width_px / 2
c_y = image_height_px / 2
K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]])
global_pcd = o3d.geometry.PointCloud()
prev_pcd = None

# Assume zero distortion
D = np.zeros((4, 1))  # Modify if distortion coefficients are known

# Create SIFT object
sift = cv2.SIFT_create(contrastThreshold=0.045)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2)

kd = []

print("Calculating keypoints and descriptors...")
for i in tqdm(range(len(image_paths))):
    img = cv2.imread(os.path.join(folder, image_paths[i]), cv2.IMREAD_GRAYSCALE)
    keypoints1, descriptors1 = sift.detectAndCompute(img, None)
    kd.append((keypoints1, descriptors1))

#pcd_matches = []
pcds = []
window_size = 1
max_pc_idx = 48

reference_R = np.eye(3)
reference_t = np.zeros((3, 1))

print("\nMatching keypoints and descriptors...")
for i in pc_list:
    #local_pcd = o3d.geometry.PointCloud()
    #inner_prev_pcd = None
    #count = 0
    
    for j in range(max(0, i-window_size), min(max_pc_idx, i+window_size+1)):
        if i != j:
            print(i, j)
            keypoints1, descriptors1 = kd[i]
            keypoints2, descriptors2 = kd[j]

            # Match descriptors
            # matches = bf.match(descriptors1, descriptors2)
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            # Sort them in the order of their distance
            # matches = sorted(matches, key=lambda x: x.distance)
            matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            #print(len(matches))
            if len(matches) > 76:

                # Extract the matched keypoints from both images
                points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
                points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

                # Assuming 'points1' and 'points2' are arrays of matched keypoints from two images
                # And 'K' is the camera matrix obtained from calibration
                E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1)
                _, R, t, mask = cv2.recoverPose(E, points1, points2, K)
                
                # Accumulate transformations relative to the reference
                R = reference_R @ R
                t = reference_R @ t + reference_t
                
                print(R, t)

                projMatr1 = np.hstack((K, np.zeros((3, 1))))
                projMatr2 = K @ np.hstack((R, t.reshape(3, 1)))

                # Triangulate points (homogeneous coordinates)
                points4D = cv2.triangulatePoints(projMatr1=projMatr1, projMatr2=projMatr2, projPoints1=points1.T,
                                                 projPoints2=points2.T)
                points3D = points4D[:3] / points4D[3]
                points3D = points3D.T

                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(points3D)
                new_temp_points = transform_points(np.asarray(temp_pcd.points), R, t)
                temp_pcd.points = o3d.utility.Vector3dVector(new_temp_points)
                
                temp_pcd, _ = temp_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)
                
                #o3d.visualization.draw_geometries([temp_pcd])
                pcds.append(temp_pcd)

def process_point_clouds(pcds, voxel_size=0.001, threshold=100, voxel_multiple = 150, iterations = 12000000, bReg = True):
    # Downsample and extract features from the point clouds
    pcd_downsampled = []
    pcd_fpfh = []
    for pcd in pcds:
        down, fpfh = preprocess_point_cloud_2(pcd, voxel_size)
        pcd_downsampled.append(down)
        pcd_fpfh.append(fpfh)

    if bReg:
        # Global registration
        result = execute_global_registration(pcd_downsampled[0], pcd_downsampled[1], pcd_fpfh[0], pcd_fpfh[1], voxel_size,
                                             voxel_multiple, iterations)
    
        # Evaluation
        evaluation = o3d.pipelines.registration.evaluate_registration(
            pcds[0], pcds[1], threshold, result.transformation)
        print(evaluation)
    
        # Update the original point clouds
        pcds[0], pcds[1] = pcd_downsampled
    
        # Refinement using ICP
        registered_images = o3d.pipelines.registration.registration_icp(
            pcds[0], pcds[1], threshold, result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(registered_images)
        print("[INFO] Transformation Matrix:")
        print(registered_images.transformation)
        draw_registration_result(pcds[0], pcds[1], registered_images)

        # Pose graph generation for multiway registration
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            pose_graph = full_registration(pcds, result)
            
    else:
        pcds[0], pcds[1] = pcd_downsampled
        result = np.identity(4)
        # Pose graph generation for multiway registration
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            pose_graph = full_registration(pcds, result)

    # Combine and visualize the transformed point clouds
    pcd_combined = o3d.geometry.PointCloud()
    for i, pcd in enumerate(pcds):
        pcd_combined += pcd.transform(pose_graph.nodes[i].pose)
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.03)
    o3d.visualization.draw_geometries([pcd_combined_down])
    print('Done!')
    
    return pcd_combined_down

final_cloud = process_point_clouds(pcds[:2], voxel_size=0.001, threshold=100, voxel_multiple=150,
                                   iterations=12000000)


# =============================================================================
# theta = np.radians(0)
# theta2 = np.radians(0)
# R = rotation_matrix('x', theta)
# pcd2_transformed = apply_rotation(pcds[5], R)
# R = rotation_matrix('y', theta2)
# pcd2_transformed = apply_rotation(pcd2_transformed, R)
# pcd = final_cloud + pcd2_transformed
# o3d.visualization.draw_geometries([pcd])
# =============================================================================


#Needs orientation work!!
final_cloud += process_point_clouds([final_cloud, pcds[3]], voxel_size=0.001, threshold=100,
                                    voxel_multiple=150, iterations=12000000, bReg=True)
o3d.visualization.draw_geometries([final_cloud])

#pcd_combined_down, _ = pcd_combined_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#o3d.visualization.draw_geometries([pcd_combined_down])
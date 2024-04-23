# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:10:35 2024

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


def preprocess_point_cloud_2(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1500000
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
        o3d.pipelines.registration.RANSACConvergenceCriteria(8000000, confidence=0.999))
    return result


def preprocess_point_cloud(pcd, voxel_size):
    # Downsample the point cloud
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    return pcd_down


def pairwise_registration(source, target):
    voxel_size = 0.01
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
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
    source_temp_2 = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # Color: Orange
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Color: Blue
    source_temp_2.paint_uniform_color([0, 1, 0])  # Color: Green
    source_temp.transform(transformation.transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp, source_temp_2])


def draw_pcds(pcd_1, pcd_2):
    pcd_1.paint_uniform_color([1, 0.706, 0])  # Color: Orange
    pcd_2.paint_uniform_color([0, 0.651, 0.929])  # Color: Blue
    o3d.visualization.draw_geometries([pcd_1, pcd_2])


# Placeholder for the image paths
folder = 'Treasure_Chest/'
images = sorted([img for img in os.listdir(folder) if img.endswith(".png")], key=numerical_sort_key)
image_paths = np.array(images)[[0, 8]]
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

# pcd_matches = []
pcds = []
window_size = 4

print("\nMatching keypoints and descriptors...")
for i in tqdm(range(len(image_paths))):
    # local_pcd = o3d.geometry.PointCloud()
    # inner_prev_pcd = None
    # count = 0
    for j in range(max(0, i - window_size), min(len(image_paths), i + window_size + 1)):
        # print(i, j)
        if i != j:
            keypoints1, descriptors1 = kd[i]
            keypoints2, descriptors2 = kd[j]

            # Match descriptors
            # matches = bf.match(descriptors1, descriptors2)
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            # Sort them in the order of their distance
            # matches = sorted(matches, key=lambda x: x.distance)
            matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            # print(len(matches))
            if len(matches) > 76:
                # img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None,
                #                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
                #
                # output_image1 = cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # output_image2 = cv2.drawKeypoints(image2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                # plt.figure(figsize=(10, 5), dpi=200)  # Adjust the size as necessary
                # plt.imshow(img_matches)
                # plt.axis('off')  # Turn off axis numbers and ticks
                # plt.title('Top 50 SIFT Matches')
                # plt.show()
                # plt.close()
                #
                # plt.figure(figsize=(10, 5), dpi=200)
                # plt.imshow(output_image1)
                # plt.show()
                # plt.close()
                #
                # plt.figure(figsize=(10, 5), dpi=200)
                # plt.imshow(output_image2)
                # plt.show()
                # plt.close()

                # Extract the matched keypoints from both images
                points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
                points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

                # Assuming 'points1' and 'points2' are arrays of matched keypoints from two images
                # And 'K' is the camera matrix obtained from calibration
                E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1)
                _, R, t, mask = cv2.recoverPose(E, points1, points2, K)

                projMatr1 = np.hstack((K, np.zeros((3, 1))))
                projMatr2 = K @ np.hstack((R, t.reshape(3, 1)))

                # Triangulate points (homogeneous coordinates)
                points4D = cv2.triangulatePoints(projMatr1=projMatr1, projMatr2=projMatr2, projPoints1=points1.T,
                                                 projPoints2=points2.T)
                points3D = points4D[:3] / points4D[3]
                points3D = points3D.T

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2])
                #
                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Y Label')
                # ax.set_zlabel('Z Label')
                # plt.show()
                # plt.close()

                # Store the pose (R, t)
                # list_of_poses.append((R, t))

                # Create Open3D point cloud from these points
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(points3D)
                # list_of_point_clouds.append(pcd)

                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(points3D)
                new_temp_points = transform_points(np.asarray(temp_pcd.points), R, t)
                temp_pcd.points = o3d.utility.Vector3dVector(new_temp_points)
                # o3d.visualization.draw_geometries([temp_pcd])
                # temp_pcd = temp_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
                pcds.append(temp_pcd)

voxel_size = 0.001
threshold = 0.001

draw_pcds(pcds[0], pcds[1])

pcd0_down, pdc0_fpfh = preprocess_point_cloud_2(pcds[0], voxel_size)
pcd1_down, pdc1_fpfh = preprocess_point_cloud_2(pcds[1], voxel_size)

result = execute_global_registration(pcd0_down, pcd1_down, pdc0_fpfh, pdc1_fpfh, voxel_size)

evaluation = o3d.pipelines.registration.evaluate_registration(pcds[0], pcds[1], threshold, result.transformation)
print(f"Evaluation: {evaluation}")

pcds[0].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcds[1].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

registered_images = o3d.pipelines.registration.registration_icp(
    pcds[0], pcds[1], threshold, result.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200000))
print(registered_images)
print("[INFO] Transformation Matrix:")
print(registered_images.transformation)
draw_registration_result(pcds[0], pcds[1], registered_images)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds)

pcd_combined = o3d.geometry.PointCloud()
pcd_combined = pcds[0].transform(pose_graph.nodes[0].pose)
pcd_combined += pcds[1].transform(pose_graph.nodes[1].pose)
pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
o3d.visualization.draw_geometries([pcd_combined_down])
print('Done!')

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:33:17 2024

@author: kevin
"""

import os
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import copy

from scipy.optimize import least_squares
from tqdm import tqdm


if os.path.exists('Treasure_Chest/plys-2') is False:
    os.makedirs('Treasure_Chest/plys-2')
ply_folder = 'Treasure_Chest/plys-2'


def numerical_sort_key(filename):
    return int(''.join(filter(str.isdigit, filename)))


def transform_points(points, R, t):
    # Apply the transformation to the points
    t = np.reshape(t, (1, 3))
    return (R @ points.T).T + t

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp_2 = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0]) # Color: Orange
    target_temp.paint_uniform_color([0, 0.651, 0.929]) # Color: Blue
    source_temp_2.paint_uniform_color([0, 1, 0]) # Color: Green
    source_temp.transform(transformation.transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp, source_temp_2])


def detect_features(img):
    keypoints1, descriptors1 = sift.detectAndCompute(img, None)
    return keypoints1, descriptors1


def match_features(kp1, des1, kp2, des2):
    matches = bf.knnMatch(des1, des2, k=2)

    # Sort them in the order of their distance
    # matches = sorted(matches, key=lambda x: x.distance)
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    # print(len(matches))
    if len(matches) > 0:

        # Extract the matched keypoints from both images
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # points1_norm, scale1, centroid1 = normalize_points(points1)
        # points2_norm, scale2, centroid2 = normalize_points(points2)
    else:
        return None, None

    return points1, points2


def refine_camera_matrix(images, K_initial):
    obj_points = []  # 3D world points
    img_points = []  # Corresponding 2D image points
    camera_matrices = []  # Camera matrices (P = K [R|t]) for each image

    for i in range(len(images) - 1):
        # Detect and match features
        kp1, des1 = detect_features(images[i])
        kp2, des2 = detect_features(images[i + 1])
        points1, points2 = match_features(kp1, des1, kp2, des2)

        if points1 is None:
            continue

        # Compute Essential Matrix
        E, mask = cv2.findEssentialMat(points1, points2, K_initial, cv2.RANSAC, 0.999, 1.0)
        _, R, t, mask = cv2.recoverPose(E, points1, points2, K_initial)

        # Triangulate Points between the first and the current image
        P1 = K_initial @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K_initial @ np.hstack((R, t))
        points4D = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points3D = points4D[:3] / points4D[3]  # Convert to non-homogeneous coordinates

        obj_points.append(points3D.T)
        img_points.append(points2)
        camera_matrices.append(P2)

    # Function to compute the total reprojection error
    def reprojection_error(params):
        fx, cx, fy, cy = params
        K_new = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        total_error = 0
        for points3D, points2D, P in zip(obj_points, img_points, camera_matrices):
            P_new = K_new @ np.linalg.inv(K_initial) @ P
            img_points_projected = cv2.projectPoints(points3D, np.eye(3), np.zeros(3), K_new, None)[0]
            # projected_norm, _, _ = normalize_points(img_points_projected.squeeze())
            # points2D_norm, _, _ = normalize_points(points2D)
            total_error += np.sum((img_points_projected.squeeze() - points2D) ** 2)
        # print(total_error)
        return total_error

    # Minimize the reprojection error
    res = least_squares(reprojection_error, [K_initial[0, 0], K_initial[0, 2], K_initial[1, 1], K_initial[1, 2]])
    f_opt, cx_opt, cy_opt = res.x
    K_optimized = np.array([[f_opt, 0, cx_opt], [0, f_opt, cy_opt], [0, 0, 1]])

    return K_optimized


# Placeholder for the image paths
folder = 'Treasure_Chest/'
images = sorted([img for img in os.listdir(folder) if img.endswith(".png")], key=numerical_sort_key)
image_paths = images
list_of_point_clouds = []
list_of_poses = []

# Camera parameters (Example values, should be obtained from calibration)
focal_length_mm = 39.14
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
D = np.zeros((4, 1))

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

pcd_matches = []
# window_size = 500
print("\nMatching keypoints and descriptors...")

for i in tqdm(range(1, len(image_paths))):
    local_pcd = o3d.geometry.PointCloud()
    inner_prev_pcd = None
    count = 0
    for j in range(0, i):
        if i == j:
            continue

        keypoints1, descriptors1 = kd[i]
        keypoints2, descriptors2 = kd[j]

        # Match descriptors
        # matches = bf.match(descriptors1, descriptors2)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Sort them in the order of their distance
        # matches = sorted(matches, key=lambda x: x.distance)
        matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(matches) < 76:
            continue
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

        i_m_a_g_e = [cv2.imread(os.path.join(folder, image_paths[i]), cv2.IMREAD_GRAYSCALE),
                        cv2.imread(os.path.join(folder, image_paths[j]), cv2.IMREAD_GRAYSCALE)]

        K = refine_camera_matrix(i_m_a_g_e, K)

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

        colors = []
        img = cv2.imread(os.path.join(folder, image_paths[i]))
        for pt in points1:
            x, y = pt
            colors.append(img[int(y), int(x)])

        colors = [[c[2] / 255, c[1] / 255, c[0] / 255] for c in colors]
        temp_pcd.colors = o3d.utility.Vector3dVector(colors[:len(temp_pcd.points)])

        if inner_prev_pcd is None:
            local_pcd = temp_pcd
            inner_prev_pcd = temp_pcd
        else:
            transformation = o3d.pipelines.registration.registration_icp(
                temp_pcd, inner_prev_pcd, max_correspondence_distance=1000,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
            temp_pcd = temp_pcd.transform(transformation.transformation)
            local_pcd = temp_pcd
            inner_prev_pcd = temp_pcd

    if prev_pcd is None:
        prev_pcd = local_pcd
    else:
        transformation = o3d.pipelines.registration.registration_icp(
            local_pcd, prev_pcd, max_correspondence_distance=100,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200000))
        local_pcd = local_pcd.transform(transformation.transformation)
        prev_pcd = local_pcd
    
    prev_pcd, _ = prev_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.2)
    if i % 15 == 0 or i == len(image_paths) - 1:
        o3d.io.write_point_cloud(f'{ply_folder}/point_cloud_{i}.ply', prev_pcd)
        o3d.visualization.draw_geometries([prev_pcd], mesh_show_wireframe=True)
    print(f"Number of points in the point cloud: {len(np.asarray(prev_pcd.points))}")


# global_cloud = combine_point_clouds(list_of_point_clouds, list_of_poses)
o3d.visualization.draw_geometries([prev_pcd], mesh_show_wireframe=True)
print("Done!")

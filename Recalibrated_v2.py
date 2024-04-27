# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 03:44:23 2024

@author: kevin
"""

import numpy as np
import cv2
import os
import copy
import open3d as o3d

from scipy.optimize import least_squares
from tqdm import tqdm

sift = cv2.SIFT_create(contrastThreshold=0.045)
# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2)


def numerical_sort_key(filename):
    return int(''.join(filter(str.isdigit, filename)))


def normalize_points(points, defined_range):
    new_max = defined_range[1]
    new_min = defined_range[0]
    old_min = np.min(points)
    old_max = np.max(points)
    normalized_arr = (points - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

    return normalized_arr


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
        fx, fy, cx, cy = params
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
    res = least_squares(reprojection_error, [K_initial[0, 0], K_initial[1, 1], K_initial[0, 2], K_initial[1, 2]])
    fx_opt, fy_opt, cx_opt, cy_opt = res.x
    K_optimized = np.array([[fx_opt, 0, 960], [0, fy_opt, 540], [0, 0, 1]])

    return K_optimized


def transform_points(points, R, t):
    # Apply the transformation to the points
    t = np.reshape(t, (1, 3))
    return (R @ points.T).T + t


focal_length_mm = 50
sensor_width_mm = 36
image_width_px = 1920
image_height_px = 1080
f_x = (focal_length_mm * image_width_px) / sensor_width_mm
f_y = (focal_length_mm * image_height_px) / sensor_width_mm
c_x = image_width_px / 2
c_y = image_height_px / 2

if os.path.exists('Treasure_Chest/plys-2') is False:
    os.makedirs('Treasure_Chest/plys-2')
ply_folder = 'Treasure_Chest/plys-2'

K_initial = np.array([[f_x, 0, c_x],
                      [0, f_y, c_y],
                      [0, 0, 1]])

# Example usage
folder = 'Treasure_Chest/'
images = []
image_path = sorted([img for img in os.listdir(folder) if img.endswith(".png")], key=numerical_sort_key)
image_idx = [35, 42, 43, 40, 41, 44, 39, 38, 45, 46, 47, 36, 37, 48, 34, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 33, 73, 32, 74, 31, 75, 76, 77, 78, 79, 80, 81, 82,
             83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             15, 16, 17, 18, 19, 20, 21, 22, 30, 29, 23, 24, 25, 26, 27, 28]

# Create a 2d array of images: image_paths where each pair is (image_idx[i]-1, image_idx[i])
image_paths = []
for i in range(len(image_idx) - 1):
    temp = [image_path[image_idx[i] - 1], image_path[image_idx[i + 1] - 1]]
    image_paths.append(temp)

kd = []
print("Calculating keypoints and descriptors...")
for i in tqdm(range(len(image_path))):
    img = cv2.imread(os.path.join(folder, image_path[i]), cv2.IMREAD_GRAYSCALE)
    keypoints1, descriptors1 = sift.detectAndCompute(img, None)
    kd.append((keypoints1, descriptors1))

Ks = []
pcds = []

x_scale = []
y_scale = []
z_scale = []

print("Processing images...")
for i in tqdm(range(len(image_paths))):
    print(f"Image Pair: {image_paths[i]}")
    i_m_a_g_e = []
    image_1 = cv2.imread(os.path.join(folder, image_paths[i][0]), cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(os.path.join(folder, image_paths[i][1]), cv2.IMREAD_GRAYSCALE)
    i_m_a_g_e.append(image_1)
    i_m_a_g_e.append(image_2)

    index1 = int(image_paths[i][0].split(".")[0][2:]) - 1
    index2 = int(image_paths[i][1].split(".")[0][2:]) - 1

    keypoints1, descriptors1 = kd[index1]
    keypoints2, descriptors2 = kd[index2]

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(matches) < 100:
        print(f"Skipping image pair {image_paths[i]} due to insufficient matches.")
        continue

    # Extract the matched keypoints from both images
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    K_refine = refine_camera_matrix(i_m_a_g_e, K_initial)
    K_initial = K_refine
    Ks.append(K_refine)

    # Assuming 'points1' and 'points2' are arrays of matched keypoints from two images
    # And 'K' is the camera matrix obtained from calibration
    E, mask = cv2.findEssentialMat(points1, points2, K_refine, method=cv2.RANSAC, prob=0.999, threshold=1)
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K_refine)

    projMatr1 = np.hstack((K_refine, np.zeros((3, 1))))
    projMatr2 = K_refine @ np.hstack((R, t.reshape(3, 1)))

    # Triangulate points (homogeneous coordinates)
    points4D = cv2.triangulatePoints(projMatr1=projMatr1, projMatr2=projMatr2, projPoints1=points1.T,
                                     projPoints2=points2.T)
    points3D = points4D[:3] / points4D[3]
    points3D = points3D.T

    temp_pcd = o3d.geometry.PointCloud()
    temp_pcd.points = o3d.utility.Vector3dVector(points3D)
    new_temp_points = transform_points(np.asarray(temp_pcd.points), R, t)
    temp_pcd.points = o3d.utility.Vector3dVector(new_temp_points)

    temp_pcd, _ = temp_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.15)

    # if not x_scale:
    #     points_of_pcd = np.asarray(temp_pcd.points)
    #     x_scale = [np.min(points_of_pcd[:, 0]), np.max(points_of_pcd[:, 0])]
    #     y_scale = [np.min(points_of_pcd[:, 1]), np.max(points_of_pcd[:, 1])]
    #     z_scale = [np.min(points_of_pcd[:, 2]), np.max(points_of_pcd[:, 2])]
    # else:
    #     points_of_pcd = np.asarray(temp_pcd.points)
    #     points_of_pcd[:, 0] = normalize_points(points_of_pcd[:, 0], x_scale)
    #     points_of_pcd[:, 1] = normalize_points(points_of_pcd[:, 1], y_scale)
    #     points_of_pcd[:, 2] = normalize_points(points_of_pcd[:, 2], z_scale)
    #     temp_pcd.points = o3d.utility.Vector3dVector(points_of_pcd)

    colors = []
    img = cv2.imread(os.path.join(folder, image_paths[i][0]))
    for pt in points1:
        x, y = pt
        colors.append(img[int(y), int(x)])

    colors = [[c[2] / 255, c[1] / 255, c[0] / 255] for c in colors]
    temp_pcd.colors = o3d.utility.Vector3dVector(colors[:len(temp_pcd.points)])

    if i % 15 == 0 or i == len(image_paths) - 1:
        o3d.io.write_point_cloud(f'{ply_folder}/point_cloud_{i}.ply', temp_pcd)
        o3d.visualization.draw_geometries([temp_pcd], mesh_show_wireframe=True)

    # o3d.visualization.draw_geometries([temp_pcd])
    pcds.append(temp_pcd)


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


def pairwise_registration(source, target, result_matrix):
    max_correspondence_distance_coarse = 100
    max_correspondence_distance_fine = 15
    print("Apply point-to-plane ICP")
    try:
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, result_matrix.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    except:
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, result_matrix,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000,
                                                              relative_rmse=1e-6,
                                                              relative_fitness=1e-6))

    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
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


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp_2 = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # Color: Orange
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Color: Blue
    source_temp_2.paint_uniform_color([0, 1, 0])  # Color: Green
    source_temp.transform(transformation.transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp, source_temp_2])
    return None


def draw_pcds(pcd_1, pcd_2):
    pcd_1.paint_uniform_color([1, 0.706, 0])  # Color: Orange
    # pcd_2.paint_uniform_color([0, 0.651, 0.929])  # Color: Blue
    o3d.visualization.draw_geometries([pcd_1, pcd_2])
    return None


def process_point_clouds(pcds, voxel_size=0.001, threshold=100, voxel_multiple=150, iterations=12000000, bReg=True):
    # Downsample and extract features from the point clouds
    pcd_downsampled = []
    pcd_fpfh = []
    for pcd in pcds:
        down, fpfh = preprocess_point_cloud_2(pcd, voxel_size)
        pcd_downsampled.append(down)
        pcd_fpfh.append(fpfh)

    if bReg:
        # Global registration
        result = execute_global_registration(pcd_downsampled[0], pcd_downsampled[1], pcd_fpfh[0], pcd_fpfh[1],
                                             voxel_size,
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


# finito_pcds = []
#
# print(f"Processing point clouds...")
# for i in tqdm(range(1, len(pcds))):
#     if i == 1:
#         pcd_first = pcds[i - 1]
#         pcd_second = pcds[i]
#     else:
#         pcd_first = finito_pcds[-1]
#         pcd_second = pcds[i]
#
#     if i % 10 == 0:
#         draw_pcds(pcd_first, pcd_second)
#
#     input_pcds = [pcd_first, pcd_second]
#
#     final_cloud = process_point_clouds(input_pcds, voxel_size=0.001, threshold=100, voxel_multiple=150,
#                                        iterations=12000000)
#     finito_pcds.append(final_cloud)
    # if i % 3 == 0:
    #     o3d.visualization.draw_geometries([pcds[i]])

# o3d.visualization.draw_geometries([finito_pcds[-1]])

# o3d.io.write_point_cloud(f'{folder}/point_cloudddd.ply', finito_pcds[-1])

# print(f"##############################################\n")
# print(f"Optimized Camera Matrices:\n")
# for i in range(len(Ks)):
#     print(f"K{i+1}:\n{Ks[i]}\n")
# print(f"##############################################\n")
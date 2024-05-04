# -*- coding: utf-8 -*-
"""
Created on Wed May  1 05:12:49 2024

@author: kevin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 05:07:12 2024

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

def optimize_distortion_params(observed_points1, observed_points2, K, dist_initial):
    
    def reprojection_error(params, normalized_points, observed_points, K):
        k1, k2 = params
        x, y = normalized_points[:, 0], normalized_points[:, 1]
        r2 = x**2 + y**2
        x_distorted = x * (1 + k1 * r2 + k2 * r2**2)
        y_distorted = y * (1 + k1 * r2 + k2 * r2**2)

        x_projected = K[0, 0] * x_distorted + K[0, 2]
        y_projected = K[1, 1] * y_distorted + K[1, 2]

        error_x = x_projected - observed_points[:, 0]
        error_y = y_projected - observed_points[:, 1]
        return np.sqrt(error_x**2 + error_y**2)

    # Normalize points using the inverse of K
    inv_K = np.linalg.inv(K)
    normalized_points1 = (inv_K @ np.hstack((observed_points1, np.ones((observed_points1.shape[0], 1)))).T).T[:, :2]

    result = least_squares(reprojection_error, [dist_initial[0], dist_initial[1]], args=(normalized_points1, observed_points2, K))
    return result.x

def refine_camera_matrix(images, K_initial, points1, points2):
    obj_points = []  # 3D world points
    img_points = []  # Corresponding 2D image points
    camera_matrices = []  # Camera matrices (P = K [R|t]) for each image

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
        f, cx, cy = params
        K_new = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        total_error = 0
        for points3D, points2D, P in zip(obj_points, img_points, camera_matrices):
            P_new = K_new @ np.linalg.inv(K_initial) @ P
            img_points_projected = cv2.projectPoints(points3D, np.eye(3), np.zeros(3), K_new, None)[0]
            total_error += np.sum((img_points_projected.squeeze() - points2D) ** 2)
        
        return total_error

    # Minimize the reprojection error
    res = least_squares(reprojection_error, [K_initial[0, 0], K_initial[0, 2], K_initial[1, 2]])
    f, cx_opt, cy_opt = res.x
    
    K_optimized = np.array([[f, 0, 960], [0, f, 540], [0, 0, 1]])

    return K_optimized


def transform_points(points, R, t):
    # Apply the transformation to the points
    t = np.reshape(t, (1, 3))
    return (R @ points.T).T + t

def normalize_open3d_point_cloud(point_cloud):
    # Convert Open3D point cloud to NumPy array
    np_points = np.asarray(point_cloud.points)
    
    if np_points.size == 0:
        raise ValueError("Input point cloud is empty.")
    
    # Compute the median of each axis
    median = np.median(np_points, axis=0)
    
    # Translate the points to center them at the origin
    centered_points = np_points - median
    
    # Calculate max and min values along each axis
    max_vals = np.max(centered_points, axis=0)
    min_vals = np.min(centered_points, axis=0)
    
    # Determine the maximum extent (the largest distance across all axes)
    max_extent = np.max(max_vals - min_vals)
    
    # Prevent division by zero (in case all points are the same or extremely close together)
    if max_extent == 0:
        raise ValueError("All points are the same or too close together to normalize.")

    # Calculate the scaling factor as the reciprocal of the maximum extent
    scale_factor = 1 / max_extent
    
    # Apply the scaling factor to the point cloud
    normalized_points = centered_points * scale_factor
    
    # Create a new Open3D point cloud object
    normalized_point_cloud = o3d.geometry.PointCloud()
    
    # Update the points of the new Open3D point cloud
    normalized_point_cloud.points = o3d.utility.Vector3dVector(normalized_points)
    
    return normalized_point_cloud



focal_length_mm = 50
sensor_width_mm = 36
image_width_px = 1920
image_height_px = 1920
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

dist_initial = [0, 0, 0, 0, 0]

# Example usage
folder = 'Treasure_Chest/'
images = []
image_path = sorted([img for img in os.listdir(folder) if img.endswith(".png")], key=numerical_sort_key)
image_idx = [35, 42, 43, 40, 41, 44, 39, 38, 45, 46, 47, 36, 37, 48, 34, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 33, 73, 32, 74, 31, 75, 76, 77, 78, 79, 80, 81, 82,
             83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             15, 16, 17, 18, 19, 20, 21, 22, 30, 29, 23, 24, 25, 26, 27, 28]

# =============================================================================
# image_idx = [35, 40, 37 ,38 ,39, 41, 36, 42, 43, 44, 45, 46, 47, 48 ,49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
#              34, 64, 65, 66, 67, 68, 69, 33, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
#              91, 92, 93, 94, 95, 96, 32, 31, 30, 29, 28, 27, 26 ,25 ,24 ,23, 22, 21, 20, 19, 18, 17, 16, 15 ,14 ,13, 12, 11, 10,
#              9, 8, 7, 6, 5, 4, 3, 2, 1]
# =============================================================================

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
ds = []
pcds = []

x_scale = []
y_scale = []
z_scale = []

print("Processing images...")
for i in tqdm(range(len(image_paths))):
    print(f"Image Pair: {image_paths[i]}")
    image_1 = cv2.imread(os.path.join(folder, image_paths[i][0]), cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(os.path.join(folder, image_paths[i][1]), cv2.IMREAD_GRAYSCALE)
    #image_1 = np.reshape(image_1, newshape=(1920, 1080))
    #image_2 = np.reshape(image_2, newshape=(1920, 1080))
    
    index1 = int(image_paths[i][0].split(".")[0][2:]) - 1
    index2 = int(image_paths[i][1].split(".")[0][2:]) - 1

    keypoints1, descriptors1 = kd[index1]
    keypoints2, descriptors2 = kd[index2]

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(matches) < 100:
    #    print(f"Skipping image pair {image_paths[i]} due to insufficient matches.")
        continue
    print(f"Pairs: {image_paths[i]}, {len(pcds)}")
    # Extract the matched keypoints from both images
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
# =============================================================================
#     dist_coeffs = optimize_distortion_params(points1, points2, K_initial, dist_initial)
#     dist_initial = [dist_coeffs[0], dist_coeffs[1], 0, 0, 0]
#     ds.append(dist_coeffs)
#     
#     points1 = cv2.undistortPoints(points1, K_initial, np.asarray(dist_initial), P=K_initial).squeeze(1)
#     points2 = cv2.undistortPoints(points2, K_initial, np.asarray(dist_initial), P=K_initial).squeeze(1)
# =============================================================================

    K_refine = refine_camera_matrix([image_1, image_2], K_initial, points1, points2)
    K_initial = K_refine
    Ks.append(K_refine)

    # Assuming 'points1' and 'points2' are arrays of matched keypoints from two images
    # And 'K' is the camera matrix obtained from calibration
    E, mask = cv2.findEssentialMat(points1, points2, K_refine, method=cv2.RANSAC, prob=0.999, threshold=1)
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K_refine)

    projMatr1 = K_refine @ np.hstack((np.eye(3), np.zeros((3, 1))))
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

    temp_pcd, _ = temp_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.01)

    colors = []
    img = cv2.imread(os.path.join(folder, image_paths[i][0]))
    #img = np.reshape(img, newshape=(1920, 1080, 3))
    for pt in points1:
        x, y = pt
        colors.append(img[int(y), int(x)])

    colors = [[c[2] / 255, c[1] / 255, c[0] / 255] for c in colors]
    temp_pcd.colors = o3d.utility.Vector3dVector(colors[:len(temp_pcd.points)])

    #temp_pcd = normalize_open3d_point_cloud(temp_pcd)

    if i % 90 == 0 or i == len(image_paths) - 1:
        o3d.io.write_point_cloud(f'{ply_folder}/point_cloud_{i}.ply', temp_pcd)
        #o3d.visualization.draw_geometries([temp_pcd], mesh_show_wireframe=True)

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
    max_correspondence_distance_coarse = 0.6
    max_correspondence_distance_fine = 0.3
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



def compute_principal_axes(pcd):
    """Compute the eigenvectors (principal axes) of the point cloud."""
    mean = np.mean(np.asarray(pcd.points), axis=0)
    cov_matrix = np.cov((np.asarray(pcd.points) - mean).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors, mean

def align_to_common_frame(pcd, target_axes):
    """Align the principal axes of a point cloud to a common set of axes."""
    source_axes, _ = compute_principal_axes(pcd)
    R = np.linalg.solve(source_axes.T, target_axes.T).T
    pcd.rotate(R, center=np.mean(np.asarray(pcd.points), axis=0))
    return pcd

def load_and_preprocess_point_cloud(pcd):
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    return pcd

def align_point_clouds_to_center(pcds, center):
    aligned_pcds = []
    for pcd in pcds:
        # Translate point cloud to center
        translation = center - np.mean(np.asarray(pcd.points), axis=0)
        pcd.translate(translation)
        # Rotate point cloud to align normals
        for point, normal in zip(pcd.points, pcd.normals):
            point_to_center = o3d.utility.Vector3dVector(np.asarray(center) - np.asarray(point))
            rotation_axis = np.cross(normal, point_to_center)
            if np.linalg.norm(rotation_axis) > 0:
                rotation_axis /= np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(normal, point_to_center) / (np.linalg.norm(normal) * np.linalg.norm(point_to_center)))
                R = pcd.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                pcd.rotate(R, center=np.asarray(point))
        aligned_pcds.append(pcd)
    return aligned_pcds

point_clouds = [copy.deepcopy(pcds[20]), copy.deepcopy(pcds[56]), copy.deepcopy(pcds[7])]
# Assuming we use the first point cloud's principal axis as the common reference
main_axes, _ = compute_principal_axes(point_clouds[0])

aligned_clouds = [align_to_common_frame(pcd, main_axes) for pcd in point_clouds]

central_point = np.array([0, 0, 0])  # Define the central point
#rotation_axis = np.array([1, 1, 0])  # Z-axis as an example, adjust as needed
aligned_clouds = align_point_clouds_to_center(aligned_clouds, central_point)

def compute_and_visualize_principal_axes(pcd):
    # Compute the mean of the point cloud
    points = np.asarray(pcd.points)
    mean = np.mean(points, axis=0)

    # Compute the covariance matrix and its eigenvalues and eigenvectors
    cov_matrix = np.cov(points - mean, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort the eigenvectors based on eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Create linesets for the principal axes
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([mean, mean + eigenvectors[:, 0] * 2, mean + eigenvectors[:, 1] * 2, mean + eigenvectors[:, 2] * 2])
    axes.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Red, Green, Blue
    return axes

#o3d.visualization.draw_geometries(aligned_clouds)

# Compute and visualize principal axes
pa1 = compute_and_visualize_principal_axes(aligned_clouds[0])
pa2 = compute_and_visualize_principal_axes(aligned_clouds[1])

#o3d.visualization.draw_geometries([pa2, pa1, aligned_clouds[0], aligned_clouds[1], aligned_clouds[2]])


def rotate_point_cloud(pcd, target_axis, angle, rotation_center):
    """Rotate the point cloud so that its major principal axis aligns with the target_axis."""
    principal_axis, _ = compute_principal_axes(pcd)
    principal_axis = principal_axis[:, 1]
    # Calculate rotation to align principal_axis with target_axis
    rotation_axis = np.cross(principal_axis, target_axis)
    if np.linalg.norm(rotation_axis) != 0:
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(principal_axis, target_axis), -1.0, 1.0))
        R = pcd.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        pcd.translate(-rotation_center)  # Center the point cloud around the rotation center
        pcd.rotate(R)
        pcd.translate(rotation_center)  # Move the point cloud back to its original position

def align_and_push_point_clouds(pcds, push_factor=1.5):
    """Align point clouds using the major principal axis of the first point cloud."""
    major_axis, center1 = compute_principal_axes(pcds[0])  # Use the first point cloud's major axis as the reference
    axis1 = major_axis[:, 1]
    
    _, center2 = compute_principal_axes(pcds[1])
    _, center3 = compute_principal_axes(pcds[2])

    centers = [center1, center2, center3]

    # Rotate each point cloud to align its major principal axis with the common axis
    for pcd, center in zip(pcds, centers):
        rotate_point_cloud(pcd, axis1, 0, center)  # Rotate each to align with the first's major axis

    # Optionally rotate around the major axis for specific arrangement
    angles = [0, 120, 240]  # Rotation angles for arrangement
    for idx, (pcd, angle, center) in enumerate(zip(pcds, angles, centers)):
        R = pcd.get_rotation_matrix_from_axis_angle(axis1 * np.radians(angle))
        pcd.rotate(R, center=center)
        _, new_center = compute_principal_axes(pcd)
        print(new_center, center)
        translation_vector = new_center - center
        arr = np.asarray(pcd.points)[0]
        if idx == 0:
            translation_vector[:] = [0.25, 0, -0.5]
        elif idx == 1:
            translation_vector[:] = [0, 0, 0]
        else:
            translation_vector[:] = [0.5, 0, -0.25]
        print(translation_vector)
        translation_vector[1] = 0  # Zero out the Z-component to ensure movement is within the XY-plane
        pcd.translate(translation_vector * push_factor)
        arr2 = np.asarray(pcd.points)[0]
        print(arr2 - arr, "Hello")
        
    return pcds

aligned_clouds = align_and_push_point_clouds(point_clouds, 10)

# Visualize the result
o3d.visualization.draw_geometries([aligned_clouds[0], aligned_clouds[1], aligned_clouds[2], pa1])
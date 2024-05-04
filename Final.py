# Imports
import numpy as np
import cv2
import os
import copy
import open3d as o3d

from scipy.optimize import least_squares
from tqdm import tqdm

# Methods

# Method to read the images from the folder and sort them
def numerical_sort_key(filename):
    return int(''.join(filter(str.isdigit, filename)))

# Method to refine the camera matrix
def refine_camera_matrix(K_initial, points1, points2):
    obj_points = []  # 3D world points
    img_points = []  # Corresponding 2D image points
    camera_matrices = []  # Camera matrices (P = K [R|t]) for each image

    # Compute Essential Matrix
    E, _ = cv2.findEssentialMat(points1, points2, K_initial, cv2.RANSAC, 0.999, 1.0)
    _, R, t, _ = cv2.recoverPose(E, points1, points2, K_initial)

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

# Method to transform the points using the given Rotation and Translation parameters
def transform_points(points, R, t):
    # Apply the transformation to the points
    t = np.reshape(t, (1, 3))
    return (R @ points.T).T + t

# Method to rotate the point clouds
def rotate_point_cloud(pc, degrees, axis='z'):
    radians = np.deg2rad(degrees)
    if axis == 'z':
        rotation_matrix = np.array([
            [np.cos(radians), -np.sin(radians), 0],
            [np.sin(radians), np.cos(radians), 0],
            [0, 0, 1]
        ])
    elif axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(radians), -np.sin(radians)],
            [0, np.sin(radians), np.cos(radians)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(radians), 0, np.sin(radians)],
            [0, 1, 0],
            [-np.sin(radians), 0, np.cos(radians)]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    # Apply the rotation matrix to the point cloud
    rotated_points = np.dot(np.asarray(pc.points), rotation_matrix.T)
    pc.points = o3d.utility.Vector3dVector(rotated_points)
    return pc

# Method to arrange the point clouds
def arrange_point_clouds(point_clouds):
    side_length = 2.75
    # Translate each point cloud to its position in the grid
    positions = [(0, 0, 0), (side_length, 0, 0), (side_length, -0.5, side_length - side_length / 2 - 0.2),
                 (side_length / 2, 0, side_length + 0.6)]

    for idx, (pc, (x, y, z)) in enumerate(zip(point_clouds, positions)):

        if idx == 0:
            pc = rotate_point_cloud(pc, degrees=62, axis='y')
            pc = rotate_point_cloud(pc, degrees=15, axis='z')
        elif idx == 1:
            pc = rotate_point_cloud(pc, degrees=-62, axis='y')
            pc = rotate_point_cloud(pc, degrees=-15, axis='z')
        elif idx == 2:
            pc = rotate_point_cloud(pc, degrees=-110, axis='y')
            pc = rotate_point_cloud(pc, degrees=-18, axis='z')
            pc = rotate_point_cloud(pc, degrees=13, axis='x')
        else:
            pc = rotate_point_cloud(pc, degrees=-5, axis='x')

        translation = np.array([x, y, z])
        pc.translate(translation, relative=False)

    return point_clouds


# Constants
# Create the sift object
sift = cv2.SIFT_create(contrastThreshold=0.045)

# Create the BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2)

focal_length_mm = 50
sensor_width_mm = 36
image_width_px = 1920
image_height_px = 1920
f_x = (focal_length_mm * image_width_px) / sensor_width_mm
f_y = (focal_length_mm * image_height_px) / sensor_width_mm
c_x = image_width_px / 2
c_y = image_height_px / 2

# Define an output folder to store the point cloud
ply_folder = 'Treasure_Chest/plys-2v2'
if os.path.exists(ply_folder) is False:
    os.makedirs(ply_folder)

# Define the initial camera matrix (K)
K_initial = np.array([[f_x, 0, c_x],
                      [0, f_y, c_y],
                      [0, 0, 1]])

# Define the input folder
folder = 'Treasure_Chest/'
images = []
image_path = sorted([img for img in os.listdir(folder) if img.endswith(".png")], key=numerical_sort_key)

# Use the output of vocabulary tree to index the matches
# Paper: https://ieeexplore.ieee.org/document/1641018
image_idx = [35, 42, 43, 40, 41, 44, 39, 38, 45, 46, 47, 36, 37, 48, 34, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 33, 73, 32, 74, 31, 75, 76, 77, 78, 79, 80, 81, 82,
             83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             15, 16, 17, 18, 19, 20, 21, 22, 30, 29, 23, 24, 25, 26, 27, 28]

image_paths = []
kd = []
Ks = []
ds = []
pcds = []

x_scale = []
y_scale = []
z_scale = []

# Create a 2d array of images: image_paths where each pair is (image_idx[i]-1, image_idx[i])
for i in range(len(image_idx) - 1):
    temp = [image_path[image_idx[i] - 1], image_path[image_idx[i + 1] - 1]]
    image_paths.append(temp)


# Logic
print("Calculating keypoints and descriptors...")
for i in tqdm(range(len(image_path))):
    img = cv2.imread(os.path.join(folder, image_path[i]), cv2.IMREAD_GRAYSCALE)
    keypoints1, descriptors1 = sift.detectAndCompute(img, None)
    kd.append((keypoints1, descriptors1))

print("Processing images...")
for i in tqdm(range(len(image_paths))):
    image_1 = cv2.imread(os.path.join(folder, image_paths[i][0]), cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(os.path.join(folder, image_paths[i][1]), cv2.IMREAD_GRAYSCALE)

    index1 = int(image_paths[i][0].split(".")[0][2:]) - 1
    index2 = int(image_paths[i][1].split(".")[0][2:]) - 1

    keypoints1, descriptors1 = kd[index1]
    keypoints2, descriptors2 = kd[index2]

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(matches) < 100:
        continue

    # Extract the matched keypoints from both images
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    K_refined = refine_camera_matrix(K_initial, points1, points2)
    K_initial = K_refined
    Ks.append(K_refined)

    # 'points1' and 'points2' are arrays of matched keypoints from two images
    # And 'K' is the camera matrix obtained from calibration
    E, _ = cv2.findEssentialMat(points1, points2, K_refined, method=cv2.RANSAC, prob=0.999, threshold=1)
    _, R, t, _ = cv2.recoverPose(E, points1, points2, K_refined)

    projMatr1 = K_refined @ np.hstack((np.eye(3), np.zeros((3, 1))))
    projMatr2 = K_refined @ np.hstack((R, t.reshape(3, 1)))

    # Triangulate points (homogeneous coordinates)
    points4D = cv2.triangulatePoints(projMatr1=projMatr1, projMatr2=projMatr2, projPoints1=points1.T,
                                     projPoints2=points2.T)
    points3D = points4D[:3] / points4D[3]
    points3D = points3D.T

    temp_pcd = o3d.geometry.PointCloud()
    temp_pcd.points = o3d.utility.Vector3dVector(points3D)
    new_temp_points = transform_points(np.asarray(temp_pcd.points), R, t)
    temp_pcd.points = o3d.utility.Vector3dVector(new_temp_points)

    temp_pcd, _ = temp_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.05)

    colors = []
    img = cv2.imread(os.path.join(folder, image_paths[i][0]))
    for pt in points1:
        x, y = pt
        colors.append(img[int(y), int(x)])

    colors = [[c[2] / 255, c[1] / 255, c[0] / 255] for c in colors]
    temp_pcd.colors = o3d.utility.Vector3dVector(colors[:len(temp_pcd.points)])

    pcds.append(temp_pcd)

# Manually arranging pointclouds for the images in Treasure_Chest
arranged_pcds = arrange_point_clouds([copy.deepcopy(pcds[0]), copy.deepcopy(pcds[18]), copy.deepcopy(pcds[52])])
o3d.visualization.draw_geometries([arranged_pcds[0], arranged_pcds[1], arranged_pcds[2]])
o3d.io.write_point_cloud(f'{ply_folder}/merged.ply', arranged_pcds[0])
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 03:44:23 2024

@author: kevin
"""

import numpy as np
import cv2
from scipy.optimize import least_squares
import os
from tqdm import tqdm

sift = cv2.SIFT_create(contrastThreshold=0.045)
# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2)

def numerical_sort_key(filename):
    return int(''.join(filter(str.isdigit, filename)))

def detect_features(img):
    keypoints1, descriptors1 = sift.detectAndCompute(img, None)
    return keypoints1, descriptors1

def match_features(kp1, des1, kp2, des2):
    matches = bf.knnMatch(des1, des2, k=2)

    # Sort them in the order of their distance
    # matches = sorted(matches, key=lambda x: x.distance)
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    #print(len(matches))
    if len(matches) > 76:

        # Extract the matched keypoints from both images
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
    return points1, points2

def refine_camera_matrix(images, K_initial):
    obj_points = []  # 3D world points
    img_points = []  # Corresponding 2D image points
    camera_matrices = []  # Camera matrices (P = K [R|t]) for each image

    for i in range(len(images)-1):
        # Detect and match features
        kp1, des1 = detect_features(images[i])
        kp2, des2 = detect_features(images[i+1])
        points1, points2 = match_features(kp1, des1, kp2, des2)

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
            total_error += np.sum((img_points_projected.squeeze() - points2D)**2)
            print(total_error)
        return total_error

    # Minimize the reprojection error
    res = least_squares(reprojection_error, [K_initial[0, 0], K_initial[0, 2], K_initial[1, 2]])
    f_opt, cx_opt, cy_opt = res.x
    K_optimized = np.array([[f_opt, 0, cx_opt], [0, f_opt, cy_opt], [0, 0, 1]])

    return K_optimized

# Example usage
folder = 'Treasure_Chest/'
images = []
image_paths = sorted([img for img in os.listdir(folder) if img.endswith(".png")], key=numerical_sort_key)
for i in tqdm(range(len(image_paths))):
    image = cv2.imread(os.path.join(folder, image_paths[i]), cv2.IMREAD_GRAYSCALE)
    images.append(image)

focal_length_mm = 50
sensor_width_mm = 36
image_width_px = 1920
image_height_px = 1080
f_x = (focal_length_mm * image_width_px) / sensor_width_mm
f_y = (focal_length_mm * image_height_px) / sensor_width_mm
c_x = image_width_px / 2
c_y = image_height_px / 2
K_initial = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]])

K_refined = refine_camera_matrix(images, K_initial)

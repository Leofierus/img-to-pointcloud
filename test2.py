# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:33:17 2024

@author: kevin
"""

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Placeholder for the image paths
image_paths = ['Frame1.png', 'Frame5.png', 'Frame10.png', 'Frame15.png',
               'Frame20.png', 'Frame25.png', 'Frame30.png', 'Frame35.png',
               'Frame40.png', 'Frame45.png', 'Frame50.png', 'Frame55.png',
               'Frame60.png', 'Frame65.png', 'Frame70.png', 'Frame75.png',
               'Frame80.png', 'Frame85.png', 'Frame90.png']  # add as many frames as needed
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

# Assume zero distortion
D = np.zeros((4, 1))  # Modify if distortion coefficients are known

for i in range(len(image_paths) - 1):
    image1 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_paths[i + 1], cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create(contrastThreshold=0.01)

    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    print(f"Number of keypoints for image1: {len(keypoints1)}")

    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    print(f"Number of keypoints for image2: {len(keypoints2)}")

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    output_image1 = cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    output_image2 = cv2.drawKeypoints(image2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(figsize=(10, 5), dpi=200)  # Adjust the size as necessary
    plt.imshow(img_matches)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.title('Top 50 SIFT Matches')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5), dpi=200)
    plt.imshow(output_image1)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5), dpi=200)
    plt.imshow(output_image2)
    plt.show()
    plt.close()

    # Extract the matched keypoints from both images
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Assuming 'points1' and 'points2' are arrays of matched keypoints from two images
    # And 'K' is the camera matrix obtained from calibration
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K)

    projMatr1 = np.hstack((K, np.zeros((3, 1))))
    projMatr2 = K @ np.hstack((R, t.reshape(3, 1)))

    # Triangulate points (homogeneous coordinates)
    points4D = cv2.triangulatePoints(projMatr1=projMatr1, projMatr2=projMatr2, projPoints1=points1.T,
                                     projPoints2=points2.T)
    points3D = points4D[:3] / points4D[3]
    points3D = points3D.T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    plt.close()

    # Store the pose (R, t)
    list_of_poses.append((R, t))

    # Create Open3D point cloud from these points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    list_of_point_clouds.append(pcd)


def transform_points(points, R, t):
    # Apply the transformation to the points
    return (R @ points.T + t).T


def combine_point_clouds(list_of_point_clouds, list_of_poses):
    global_point_cloud = o3d.geometry.PointCloud()

    for point_cloud, (R, t) in zip(list_of_point_clouds, list_of_poses):
        # Assuming each point_cloud is an instance of o3d.geometry.PointCloud
        # And each pose is a tuple (R, t), where R is a rotation matrix and t is a translation vector

        # Convert Open3D point cloud to numpy array
        current_points = np.asarray(point_cloud.points)

        # Transform points
        transformed_points = transform_points(current_points, R, t)

        # Create a new point cloud from transformed points
        transformed_point_cloud = o3d.geometry.PointCloud()
        transformed_point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

        # Merge the transformed point cloud into the global point cloud
        global_point_cloud += transformed_point_cloud

    # Optionally: remove duplicates and outliers
    # global_point_cloud = global_point_cloud.voxel_down_sample(voxel_size=0.05)
    # global_point_cloud, ind = global_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return global_point_cloud


global_cloud = combine_point_clouds(list_of_point_clouds, list_of_poses)
o3d.visualization.draw_geometries([global_cloud])
print("Done!")

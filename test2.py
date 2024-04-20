# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:33:17 2024

@author: kevin
"""

import os
from tqdm import tqdm
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import plotly.graph_objs as go


def numerical_sort_key(filename):
    return int(''.join(filter(str.isdigit, filename)))


def transform_points(points, R, t):
    # Apply the transformation to the points
    t = np.reshape(t, (1, 3))
    return (R @ points.T).T + t


# Placeholder for the image paths
folder = 'Treasure_Chest/'
images = sorted([img for img in os.listdir(folder) if img.endswith(".png")], key=numerical_sort_key)
image_paths = images[30:60]
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
sift = cv2.SIFT_create(contrastThreshold=0.01)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2)

kd = []

print("Calculating keypoints and descriptors...")
for i in tqdm(range(len(image_paths))):
    img = cv2.imread(os.path.join(folder, image_paths[i]), cv2.IMREAD_GRAYSCALE)
    keypoints1, descriptors1 = sift.detectAndCompute(img, None)
    kd.append((keypoints1, descriptors1))

pcd_matches = []
print("\nMatching keypoints and descriptors...")
for i in tqdm(range(len(image_paths))):
    local_pcd = o3d.geometry.PointCloud()
    inner_prev_pcd = None
    count = 0
    for j in range(len(image_paths)):
        if i != j:
            keypoints1, descriptors1 = kd[i]
            keypoints2, descriptors2 = kd[j]

            # Match descriptors
            # matches = bf.match(descriptors1, descriptors2)
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            # Sort them in the order of their distance
            # matches = sorted(matches, key=lambda x: x.distance)
            matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

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

                if inner_prev_pcd is None:
                    local_pcd = temp_pcd
                    inner_prev_pcd = temp_pcd
                else:
                    transformation = o3d.pipelines.registration.registration_icp(
                        temp_pcd, inner_prev_pcd, max_correspondence_distance=50,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
                    temp_pcd.transform(transformation.transformation)
                    local_pcd = temp_pcd
                    inner_prev_pcd = temp_pcd

                # print(f"Processed images {i} and {j}")
                count += 1
                pcd_matches.append(count)
    # print(f"Processed {count} images")

    # local_pcd = local_pcd.voxel_down_sample(voxel_size=0.05)

    if prev_pcd is not None:
        transformation = o3d.pipelines.registration.registration_icp(
            local_pcd, prev_pcd, max_correspondence_distance=50,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        local_pcd.transform(transformation.transformation)

    global_pcd = local_pcd
    prev_pcd = local_pcd

    global_pcd, _ = global_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)


for i in range(len(pcd_matches)):
    print(f"Processed images {i}, {pcd_matches[i]}")
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(list_of_point_clouds)):
#     ax.scatter(np.asarray(list_of_point_clouds[i].points)[:, 0],
#                np.asarray(list_of_point_clouds[i].points)[:, 1],
#                np.asarray(list_of_point_clouds[i].points)[:, 2])
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#
# plt.show()
# plt.close()

# Create an empty list to store traces for each point cloud
# traces = []
#
# # Iterate over each point cloud
# for i, point_cloud in enumerate(list_of_point_clouds):
#     # Extract x, y, and z coordinates from the point cloud
#     x = np.asarray(point_cloud.points)[:, 0]
#     y = np.asarray(point_cloud.points)[:, 1]
#     z = np.asarray(point_cloud.points)[:, 2]
#
#     # Create a scatter trace for the current point cloud
#     trace = go.Scatter3d(
#         x=x,
#         y=y,
#         z=z,
#         mode='markers',
#         marker=dict(
#             size=3,
#             color='rgb(255, 0, 0)',  # You can specify the color here
#             opacity=0.8
#         ),
#         name=f'Point Cloud {i+1}'
#     )
#
#     # Append the trace to the list of traces
#     traces.append(trace)
#
# # Create the layout for the plot
# layout = go.Layout(
#     scene=dict(
#         xaxis=dict(title='X Label'),
#         yaxis=dict(title='Y Label'),
#         zaxis=dict(title='Z Label')
#     )
# )
#
# # Create the figure
# fig = go.Figure(data=traces, layout=layout)
#
# # Show the interactive plot
# fig.show()


def normalize(points3D):
    # Normalize the points3D
    points3D[:, 0] = (points3D[:, 0] - np.min(points3D[:, 0])) / (
                np.max(points3D[:, 0]) - np.min(points3D[:, 0])) * 1000
    points3D[:, 1] = (points3D[:, 1] - np.min(points3D[:, 1])) / (
                np.max(points3D[:, 1]) - np.min(points3D[:, 1])) * 1000
    points3D[:, 2] = (points3D[:, 2] - np.min(points3D[:, 2])) / (
                np.max(points3D[:, 2]) - np.min(points3D[:, 2])) * 1000

    return points3D


def combine_point_clouds(list_of_point_clouds, list_of_poses):
    global_point_cloud = o3d.geometry.PointCloud()

    for point_cloud, (R, t) in zip(list_of_point_clouds, list_of_poses):
        # Assuming each point_cloud is an instance of o3d.geometry.PointCloud
        # And each pose is a tuple (R, t), where R is a rotation matrix and t is a translation vector

        # Convert Open3D point cloud to numpy array
        current_points = np.asarray(point_cloud.points)

        # Transform points
        transformed_points = transform_points(current_points, R, t)
        #transformed_points = normalize(transformed_points)

        # Create a new point cloud from transformed points
        transformed_point_cloud = o3d.geometry.PointCloud()
        transformed_point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

        # Merge the transformed point cloud into the global point cloud
        if not global_point_cloud.has_points():
            global_point_cloud = transformed_point_cloud
        else:
            global_points = np.vstack((np.asarray(global_point_cloud.points), transformed_points))
            global_point_cloud.points = o3d.utility.Vector3dVector(global_points)

    # Optionally: remove duplicates and outliers
    global_point_cloud = global_point_cloud.voxel_down_sample(voxel_size=0.05)
    global_point_cloud, ind = global_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return global_point_cloud


# global_cloud = combine_point_clouds(list_of_point_clouds, list_of_poses)
o3d.visualization.draw_geometries([global_pcd])
print("Done!")

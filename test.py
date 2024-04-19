# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:58:15 2024

@author: kevin
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread('Frame1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('Frame5.png', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create(contrastThreshold=0.04)

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

img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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




# Parameters
focal_length_mm = 39.14  # example value in millimeters
sensor_width_mm = 36  # example sensor width in millimeters
image_width_px = 1920  # resolution width in pixels
image_height_px = 1080  # resolution height in pixels

# Calculate fx, fy
f_x = (focal_length_mm * image_width_px) / sensor_width_mm
f_y = (focal_length_mm * image_height_px) / sensor_width_mm

# Calculate cx, cy
c_x = image_width_px / 2
c_y = image_height_px / 2

# Assemble the intrinsic matrix K
K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]])
print("Intrinsic Matrix K:")
print(K)

# Extract the matched keypoints from both images
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])


# Assuming 'points1' and 'points2' are arrays of matched keypoints from two images
# And 'K' is the camera matrix obtained from calibration
E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask = cv2.recoverPose(E, points1, points2, K)

projMatr1 = np.hstack((K, np.zeros((3, 1))))
projMatr2 = np.hstack((R, t))


# Triangulate points (homogeneous coordinates)
points4D = cv2.triangulatePoints(projMatr1=projMatr1, projMatr2=projMatr2, projPoints1=points1.T, projPoints2=points2.T)
points3D = points4D[:3] / points4D[3]
points3D = points3D.T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

import open3d as o3d

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points3D)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
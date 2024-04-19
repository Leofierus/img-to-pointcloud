import os
import cv2
import numpy as np
import open3d as o3d

from tqdm import tqdm


def sift_detector(image_path):
    img = cv2.imread(image_path)
    # image_path = image_path.replace('Final Project/', 'Final Project/sift-2/')
    # if os.path.exists('Final Project/sift-2') is False:
    #     os.makedirs('Final Project/sift-2')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    kp_np = np.array([k.pt for k in kp])

    # image = cv2.drawKeypoints(gray, kp, img)
    # cv2.imwrite(image_path, image)

    return kp_np, des, gray, img


def sift_matcher(image1, image2):
    kp1, des1, gray1, img1 = sift_detector(f'Final Project/{image1}')
    kp2, des2, gray2, img2 = sift_detector(f'Final Project/{image2}')
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # # Convert numpy arrays of keypoints to list of cv2.KeyPoint objects
    # kp1_cv2 = [cv2.KeyPoint(x, y, size=1) for x, y in kp1]
    # kp2_cv2 = [cv2.KeyPoint(x, y, size=1) for x, y in kp2]
    #
    # img3 = cv2.drawMatchesKnn(gray1, kp1_cv2, gray2, kp2_cv2, good,
    #                           None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imwrite(f'Final Project/sift-2/matches/{image1}_{image2}.png', img3)
    # print(f"Matches saved for {image1} and {image2}")

    print(f"Number of good matches: {len(good)}")

    src_pts = np.float32([kp1[m.queryIdx] for match in good for m in match]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx] for match in good for m in match]).reshape(-1, 1, 2)

    E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=39.14, pp=(1920/2, 1080/2)
                                   , method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, focal=39.14, pp=(1920/2, 1080/2))

    return None, None


def triangulate_points(kp1, kp2, R, t):
    if len(kp1) == 0 or len(kp2) == 0:
        return np.array([])

    min_num_keypoints = min(len(kp1), len(kp2))
    if min_num_keypoints < 4:
        return np.array([])  # Not enough keypoints for triangulation

    kp1 = kp1[:min_num_keypoints]  # Trim keypoints to have the same number
    kp2 = kp2[:min_num_keypoints]

    K = np.array([[39.14, 0, 36/2], [0, 39.14, 36/2], [0, 0, 1]])  # Camera intrinsics

    # Projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    R2 = np.matmul(R, np.eye(3))
    t2 = t.reshape(3, 1)
    P2 = np.hstack((R2, t2))

    # Triangulate points
    points_4d_homogeneous = cv2.triangulatePoints(P1, P2, kp1.T, kp2.T)
    points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]

    return points_3d.T


def numerical_sort_key(filename):
    return int(''.join(filter(str.isdigit, filename)))


def main():
    try:
        folder = 'Final Project'
        images = sorted([img for img in os.listdir(folder) if img.endswith(".png")], key=numerical_sort_key)
        poses = {}
        print("Images scanned successfully")

        for i in tqdm(range(len(images))):
            j = i + 1 if i + 1 < len(images) else 0
            R, t = sift_matcher(images[i], images[j])
            poses[(i, j)] = (R, t)
        print("Poses calculated successfully")

        # Triangulate points
        triangulated_points = []
        for (i, j), (R, t) in tqdm(poses.items()):
            kp1, des1, gray1, img1 = sift_detector(f'Final Project/{images[i]}')
            kp2, des2, gray2, img2 = sift_detector(f'Final Project/{images[j]}')
            points_3d = triangulate_points(kp1, kp2, R, t)
            triangulated_points.append(points_3d)
        print("Triangulated points successfully")

        # Concatenate all points into a single array
        point_cloud = np.concatenate(triangulated_points, axis=0)

        # Convert numpy array to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        print("Point cloud created successfully")

        # Visualize point cloud
        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(folder + "/point_cloud.ply", pcd)

    except Exception as e:
        print(f'Error: {e}')
        print(f"Stack trace: {e.with_traceback()}")


if __name__ == "__main__":
    main()

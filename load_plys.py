import os
from tqdm import tqdm
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import copy

ply_folder = 'Manual'
ply_files = os.listdir(ply_folder)
ply_files = [f for f in ply_files if f.endswith('.ply')  and f != 'merged.ply']

# ply_folder = 'Treasure_Chest/plys'
# ply_files = ['merged.ply']

# Load all ply files
plys = []

# Generate random colors for each ply file
# colors = np.random.rand(len(ply_files), 3)
for i in tqdm(range(len(ply_files))):
    ply = o3d.io.read_point_cloud(os.path.join(ply_folder, ply_files[i]))
    # ply.paint_uniform_color(colors[i])
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(ply.cluster_dbscan(eps=0.0002, min_points=5, print_progress=True))
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # ply.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([ply])
    plys.append(ply)

# Visualize all ply files
o3d.visualization.draw_geometries(plys)

#
# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     source_temp_2 = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0.706, 0])  # Color: Orange
#     target_temp.paint_uniform_color([0, 0.651, 0.929])  # Color: Blue
#     source_temp_2.paint_uniform_color([0, 1, 0])  # Color: Green
#     source_temp.transform(transformation.transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp, source_temp_2])
#
#
# def preprocess_point_cloud(pcd, voxel_size):
#     print(":: Downsample with a voxel size %.3f." % voxel_size)
#     pcd_down = pcd.voxel_down_sample(voxel_size)
#
#     radius_normal = voxel_size * 2
#     print(":: Estimate normal with search radius %.3f." % radius_normal)
#     pcd_down.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
#
#     radius_feature = voxel_size * 5
#     print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     return pcd_down, pcd_fpfh
#
#
# def execute_global_registration(source_down, target_down, source_fpfh,
#                                 target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 100000000
#     print(":: RANSAC registration on downsampled point clouds.")
#     print("   Since the downsampling voxel size is %.3f," % voxel_size)
#     print("   we use a liberal distance threshold %.3f." % distance_threshold)
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, True,
#         distance_threshold,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         3, [
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
#                 0.9),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#                 distance_threshold)
#         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
#     return result
#
#
# def refine_registration(source, target, result_ransac, voxel_size):
#     distance_threshold = voxel_size * 100000000
#     print(":: Point-to-plane ICP registration is applied on original point")
#     print("   clouds to refine the alignment. This time we use a strict")
#     print("   distance threshold %.3f." % distance_threshold)
#     result = o3d.pipelines.registration.registration_icp(
#         source, target, distance_threshold, result_ransac.transformation,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
#     return result
#
#
# # Define the global registration method
# def global_registration(plys_array):
#     final_ply = copy.deepcopy(plys_array[0])
#     for ply_val in tqdm(range(1, len(plys_array))):
#         pcd_1 = copy.deepcopy(plys_array[ply_val - 1])
#         pcd_2 = copy.deepcopy(plys_array[ply_val])
#
#         voxel_size = 0.001
#
#         source_down, source_fpfh = preprocess_point_cloud(pcd_1, voxel_size)
#         target_down, target_fpfh = preprocess_point_cloud(pcd_2, voxel_size)
#
#         result_ransac = execute_global_registration(source_down, target_down,
#                                                     source_fpfh, target_fpfh,
#                                                     voxel_size)
#
#         # draw_registration_result(source_down, target_down, result_ransac)
#
#         result_icp = refine_registration(pcd_1, pcd_2, result_ransac, voxel_size)
#
#         # draw_registration_result(pcd_1, pcd_2, result_icp)
#
#         final_ply = final_ply + pcd_2.transform(result_icp.transformation)
#
#     return final_ply
#
#
# # Perform global registration
# result = global_registration(plys)
#
# # Visualize the aligned ply files
# o3d.visualization.draw_geometries([result])

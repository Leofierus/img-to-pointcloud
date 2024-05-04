import os
from tqdm import tqdm
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import copy

ply_folder = 'Treasure_Chest/plys-2v2'
ply_files = os.listdir(ply_folder)
ply_files = [f for f in ply_files if f.endswith('.ply') and f == 'merged.ply']

# ply_folder = 'Treasure_Chest/plys'
# ply_files = ['merged.ply']

# Load all ply files
plys = []

# Generate random colors for each ply file
# colors = np.random.rand(len(ply_files), 3)
for i in tqdm(range(len(ply_files))):
    if i==1:
        continue
    ply = o3d.io.read_point_cloud(os.path.join(ply_folder, ply_files[i]))
    # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(ply)
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(ply, 0.001, tetra_mesh, pt_map)
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(ply, 0.01)
    # mesh_2 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     ply, o3d.utility.DoubleVector([0.005, 0.01, 0.02, 0.04]))
    # ply.paint_uniform_color(colors[i])
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(ply.cluster_dbscan(eps=0.0002, min_points=5, print_progress=True))
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # ply.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([ply])
    # o3d.visualization.draw_geometries([ply, mesh])
    # o3d.visualization.draw_geometries([ply, mesh_2])
    plys.append(ply)

# # Visualize all ply files
o3d.visualization.draw_geometries(plys)
#
# def preprocess_point_cloud_2(pcd, voxel_size):
#     print(":: Downsample with a voxel size %.3f." % voxel_size)
#     pcd_down = pcd.voxel_down_sample(voxel_size)
#     radius_normal = voxel_size * 150
#     pcd_down.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
#     radius_feature = voxel_size * 200
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     return pcd_down, pcd_fpfh
#
#
# def execute_global_registration(source_down, target_down, source_fpfh,
#                                 target_fpfh, voxel_size, voxel_multiple, iterations):
#     distance_threshold = voxel_size * voxel_multiple
#     print(":: RANSAC registration on downsampled point clouds.")
#     print("   Since the downsampling voxel size is %.3f," % voxel_size)
#     print("   we use a liberal distance threshold %.3f." % distance_threshold)
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, True,
#         distance_threshold,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         4,  # RANSAC n points
#         [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
#         o3d.pipelines.registration.RANSACConvergenceCriteria(iterations, confidence=1))
#     return result
#
#
# def pairwise_registration(source, target, result_matrix):
#     max_correspondence_distance_coarse = 100
#     max_correspondence_distance_fine = 15
#     print("Apply point-to-plane ICP")
#     try:
#         icp_coarse = o3d.pipelines.registration.registration_icp(
#             source, target, max_correspondence_distance_coarse, result_matrix.transformation,
#             o3d.pipelines.registration.TransformationEstimationPointToPlane())
#     except:
#         icp_coarse = o3d.pipelines.registration.registration_icp(
#             source, target, max_correspondence_distance_coarse, result_matrix,
#             o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#             o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000,
#                                                               relative_rmse=1e-6,
#                                                               relative_fitness=1e-6))
#
#     icp_fine = o3d.pipelines.registration.registration_icp(
#         source, target, max_correspondence_distance_fine,
#         icp_coarse.transformation,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane())
#     transformation_icp = icp_fine.transformation
#     information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
#         source, target, max_correspondence_distance_fine,
#         icp_fine.transformation)
#     return transformation_icp, information_icp
#
#
# def full_registration(pcds, result_matrix):
#     print(len(pcds))
#     pose_graph = o3d.pipelines.registration.PoseGraph()
#     odometry = np.identity(4)
#     pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
#     n_pcds = len(pcds)
#     for source_id in range(n_pcds):
#         for target_id in range(source_id + 1, n_pcds):
#             transformation_icp, information_icp = pairwise_registration(
#                 pcds[source_id], pcds[target_id], result_matrix)
#             print("Build o3d.registration.PoseGraph")
#             if target_id == source_id + 1:  # odometry case
#                 print("Odometry")
#                 odometry = np.dot(transformation_icp, odometry)
#                 pose_graph.nodes.append(
#                     o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
#                 pose_graph.edges.append(
#                     o3d.pipelines.registration.PoseGraphEdge(source_id,
#                                                              target_id,
#                                                              transformation_icp,
#                                                              information_icp,
#                                                              uncertain=False))
#             else:  # loop closure case
#                 pose_graph.edges.append(
#                     o3d.pipelines.registration.PoseGraphEdge(source_id,
#                                                              target_id,
#                                                              transformation_icp,
#                                                              information_icp,
#                                                              uncertain=True))
#     return pose_graph
#
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
#     return None
#
#
# def draw_pcds(pcd_1, pcd_2):
#     pcd_1.paint_uniform_color([1, 0.706, 0])  # Color: Orange
#     # pcd_2.paint_uniform_color([0, 0.651, 0.929])  # Color: Blue
#     o3d.visualization.draw_geometries([pcd_1, pcd_2])
#     return None
#
#
# def process_point_clouds(pcds, voxel_size=0.001, threshold=100, voxel_multiple=150, iterations=12000000, bReg=False):
#     # Downsample and extract features from the point clouds
#     pcd_downsampled = []
#     pcd_fpfh = []
#     for pcd in pcds:
#         down, fpfh = preprocess_point_cloud_2(pcd, voxel_size)
#         pcd_downsampled.append(down)
#         pcd_fpfh.append(fpfh)
#
#     if bReg:
#         # Global registration
#         result = execute_global_registration(pcd_downsampled[0], pcd_downsampled[1], pcd_fpfh[0], pcd_fpfh[1],
#                                              voxel_size,
#                                              voxel_multiple, iterations)
#
#         # Evaluation
#         evaluation = o3d.pipelines.registration.evaluate_registration(
#             pcds[0], pcds[1], threshold, result.transformation)
#         print(evaluation)
#
#         # Update the original point clouds
#         pcds[0], pcds[1] = pcd_downsampled
#
#         # Refinement using ICP
#         registered_images = o3d.pipelines.registration.registration_icp(
#             pcds[0], pcds[1], threshold, result.transformation,
#             o3d.pipelines.registration.TransformationEstimationPointToPlane())
#         # print(registered_images)
#         print("[INFO] Transformation Matrix:")
#         print(registered_images.transformation)
#         draw_registration_result(pcds[0], pcds[1], registered_images)
#
#         # Pose graph generation for multiway registration
#         with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#             pose_graph = full_registration(pcds, result)
#
#     else:
#         pcds[0], pcds[1] = pcd_downsampled
#         result = np.identity(4)
#         # Pose graph generation for multiway registration
#         with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#             pose_graph = full_registration(pcds, result)
#
#     # Combine and visualize the transformed point clouds
#     pcd_combined = o3d.geometry.PointCloud()
#     for i, pcd in enumerate(pcds):
#         pcd_combined += pcd.transform(pose_graph.nodes[i].pose)
#     pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.03)
#     o3d.visualization.draw_geometries([pcd_combined_down])
#     print('Done!')
#
#     return pcd_combined_down
#
#
# finito_pcds = []
# print(f"Processing point clouds...")
# for i in tqdm(range(1, len(plys))):
#     # if i == 1:
#     pcd_first = plys[i - 1]
#     pcd_second = plys[i]
#     # else:
#     #     pcd_first = finito_pcds[-1]
#     #     pcd_second = plys[i]
#
#     draw_pcds(pcd_first, pcd_second)
#
#     input_pcds = [pcd_first, pcd_second]
#
#     final_cloud = process_point_clouds(input_pcds, voxel_size=0.001, threshold=100, voxel_multiple=150,
#                                        iterations=12000000)
#     finito_pcds.append(final_cloud)

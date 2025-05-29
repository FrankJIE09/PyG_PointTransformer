# estimate_pose_from_h5_cmdline.py
# 加载 HDF5 样本，执行语义分割+聚类，然后对每个实例进行可选的粗匹配 (RANSAC/FGR) + 精匹配 (ICP) 姿态估计。
# 实例点云不再进行额外的预处理步骤。

import sys
import os
import datetime
import copy
import h5py
import numpy as np
import torch
import argparse
import time
from scipy.spatial.transform import Rotation as R

# --- 导入本地模块 ---
try:
    # Ensure your model.py is in the same directory or PYTHONPATH
    from model import PyG_PointTransformerSegModel  # 语义分割模型类
except ImportError as e:
    print(f"FATAL Error importing model: {e}")
    print("Please ensure 'model.py' is in the current directory or your PYTHONPATH.")
    sys.exit(1)

# --- 导入 Open3D and Scikit-learn ---
try:
    import open3d as o3d
    from open3d.pipelines import registration as o3d_reg

    OPEN3D_AVAILABLE = True
except ImportError:
    print("FATAL Error: Open3D not found. Please install Open3D (pip install open3d).")
    OPEN3D_AVAILABLE = False
    sys.exit(1)  # Open3D is critical
try:
    from sklearn.cluster import DBSCAN

    SKLEARN_AVAILABLE = True
except ImportError:
    print("FATAL Error: scikit-learn not found. Please install scikit-learn (pip install scikit-learn).")
    SKLEARN_AVAILABLE = False
    sys.exit(1)  # Sklearn is critical


# --- 可视化函数: 显示单个点云 ---
def visualize_single_pcd(pcd, window_name="Point Cloud", point_color=[0.5, 0.5, 0.5]):
    if not OPEN3D_AVAILABLE: return
    if not pcd.has_points():
        print(f"Skipping visualization for {window_name}: No points to display.")
        return

    pcd_vis = copy.deepcopy(pcd)
    if not pcd_vis.has_colors():
        pcd_vis.paint_uniform_color(point_color)

    print(f"\nDisplaying Point Cloud: {window_name} ({len(pcd_vis.points)} points)")
    print("(Close the window to continue...)")
    try:
        o3d.visualization.draw_geometries([pcd_vis], window_name=window_name)
        print(f"Visualization window '{window_name}' closed.")
    except Exception as e:
        print(f"Error during visualization of '{window_name}': {e}")


# --- 可视化函数: 显示中间对齐结果 (例如粗匹配的降采样/居中点云) ---
def visualize_intermediate_alignment(source_pcd_aligned, target_pcd, window_name="Intermediate Alignment",
                                     source_color=[1, 0.706, 0], target_color=[0, 0.651, 0.929],
                                     source_name="Aligned Source (Downsampled/Centered)",
                                     target_name="Target (Downsampled/Centered)"):
    if not OPEN3D_AVAILABLE: return
    source_vis = copy.deepcopy(source_pcd_aligned)
    target_vis = copy.deepcopy(target_pcd)
    source_vis.paint_uniform_color(source_color)
    target_vis.paint_uniform_color(target_color)
    print(f"\nDisplaying Alignment: {window_name}...")
    print(f"Color 1 ({source_name}): {source_color} | Color 2 ({target_name}): {target_color}")
    print("(Close the window to continue...)")
    try:
        o3d.visualization.draw_geometries([source_vis, target_vis], window_name=window_name)
        print(f"Alignment visualization window '{window_name}' closed.")
    except Exception as e:
        print(f"Error during visualization of '{window_name}': {e}")


# --- 可视化函数: 显示最终姿态或粗略姿态对齐结果 (变换后的模型 vs 场景实例) ---
def visualize_pose_alignment_of_model_to_instance(source_pcd_transformed_model, instance_pcd_observed,
                                                  window_name="Pose Alignment Result",
                                                  model_name_in_legend="CAD Model transformed by pose"):
    if not OPEN3D_AVAILABLE: return
    instance_pcd_vis = copy.deepcopy(instance_pcd_observed)
    transformed_model_vis = copy.deepcopy(source_pcd_transformed_model)
    transformed_model_vis.paint_uniform_color([1, 0.706, 0])  # Yellow for transformed model
    instance_pcd_vis.paint_uniform_color([0, 0.651, 0.929])  # Blue for observed instance
    print(f"\nDisplaying Pose Alignment: {window_name}...")
    print(f"Yellow: {model_name_in_legend} | Blue: Observed Instance in Scene")
    print("(Close the window to continue...)")
    try:
        o3d.visualization.draw_geometries([transformed_model_vis, instance_pcd_vis], window_name=window_name)
        print(f"Alignment visualization window '{window_name}' closed.")
    except Exception as e:
        print(f"Error during visualization: {e}")


# --- 可视化函数: 显示变换后的模型在原始场景中 ---
def visualize_transformed_model_in_scene(original_scene_pcd, target_model_geometry, estimated_pose,
                                         window_name="Transformed Model in Original Scene"):
    if not OPEN3D_AVAILABLE: return
    scene_vis = copy.deepcopy(original_scene_pcd)
    model_vis = copy.deepcopy(target_model_geometry)
    model_vis.transform(estimated_pose)
    if not scene_vis.has_colors():
        scene_vis.paint_uniform_color([0.8, 0.8, 0.8])
    if isinstance(model_vis, o3d.geometry.TriangleMesh):
        model_vis.paint_uniform_color([0, 1, 0])  # Green
        if not model_vis.has_vertex_normals(): model_vis.compute_vertex_normals()
    elif isinstance(model_vis, o3d.geometry.PointCloud):
        model_vis.paint_uniform_color([0, 1, 0])  # Green
    print(f"\nDisplaying: {window_name}")
    print("Scene Color: Original Scene | Green: Transformed Target Model")
    print("(Close the window to continue...)")
    try:
        geometries_to_draw = [scene_vis]
        if (isinstance(model_vis, o3d.geometry.TriangleMesh) and model_vis.has_vertices()) or \
                (isinstance(model_vis, o3d.geometry.PointCloud) and model_vis.has_points()):
            geometries_to_draw.append(model_vis)
        else:
            print(f"Warning: Transformed model for '{window_name}' has no geometry to draw.")
        if len(geometries_to_draw) > 1:
            o3d.visualization.draw_geometries(geometries_to_draw, window_name=window_name)
            print(f"Visualization window '{window_name}' closed.")
        else:
            print(f"Not enough geometries to display for '{window_name}'.")
    except Exception as e:
        print(f"Error during visualization of '{window_name}': {e}")


# --- FPFH 特征计算辅助函数 ---
def preprocess_point_cloud_for_fpfh(pcd, voxel_size, normal_radius_factor, feature_radius_factor, min_points,
                                    name="Point Cloud"):
    if not OPEN3D_AVAILABLE: return None, None

    # print(f"    FPFH Preprocessing for {name}: Original points {len(pcd.points)}") # Verbose
    if len(pcd.points) < min_points:
        print(
            f"    FPFH Preprocessing for {name}: Too few points ({len(pcd.points)} < {min_points}) before downsampling. Skipping FPFH.")
        return None, None

    pcd_down = pcd.voxel_down_sample(voxel_size)
    # print(
    #     f"    FPFH Preprocessing for {name}: Downsampled to {len(pcd_down.points)} points using voxel size {voxel_size:.4f}") # Verbose
    if not pcd_down.has_points() or len(pcd_down.points) < min_points:
        print(
            f"    FPFH Preprocessing for {name}: Not enough points after downsampling ({len(pcd_down.points)} < {min_points}). Cannot compute FPFH.")
        return None, None

    radius_normal = voxel_size * normal_radius_factor
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    if not pcd_down.has_normals():
        print(f"    FPFH Preprocessing for {name}: Failed to estimate normals.")
        return None, None
    # print(f"    FPFH Preprocessing for {name}: Normals estimated with radius {radius_normal:.4f}") # Verbose

    radius_feature = voxel_size * feature_radius_factor
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    # print(f"    FPFH Preprocessing for {name}: FPFH features computed with radius {radius_feature:.4f}") # Verbose
    return pcd_down, pcd_fpfh


# --- 主函数 ---
def main(args):
    print(f"Starting Pose Estimation from HDF5 at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- 加载 HDF5 数据 ---
    print(f"Loading data from HDF5: {args.input_h5}, Sample: {args.sample_index}")
    points_original_np, points_rgb_np, model_uses_rgb = None, None, False
    if not os.path.exists(args.input_h5): print(f"FATAL: HDF5 file not found: {args.input_h5}"); sys.exit(1)
    try:
        with h5py.File(args.input_h5, 'r') as f:
            dset_data = f['data'];
            dset_rgb = f.get('rgb', None)
            points_original_np = dset_data[args.sample_index].astype(np.float32)
            if dset_rgb is not None and not args.no_rgb and dset_rgb.shape[0] > args.sample_index and \
                    dset_rgb[args.sample_index].shape[0] == points_original_np.shape[0]:
                points_rgb_np = dset_rgb[args.sample_index].astype(np.uint8);
                model_uses_rgb = True;
                print("RGB data loaded.")
            else:
                print("RGB data not found/enabled or mismatched. Using XYZ only.")
    except Exception as e:
        print(f"FATAL Error loading HDF5: {e}");
        sys.exit(1)
    if points_original_np is None: print("FATAL: Points data could not be loaded."); sys.exit(1)

    original_scene_o3d_pcd = o3d.geometry.PointCloud()
    original_scene_o3d_pcd.points = o3d.utility.Vector3dVector(points_original_np)
    if points_rgb_np is not None and model_uses_rgb:
        original_scene_o3d_pcd.colors = o3d.utility.Vector3dVector(points_rgb_np.astype(np.float64) / 255.0)

    # --- 加载语义模型 ---
    print(f"\nLoading SEMANTIC model: {args.checkpoint}")
    if not os.path.exists(args.checkpoint): print(f"FATAL: Checkpoint not found: {args.checkpoint}"); sys.exit(1)
    current_sample_uses_rgb = (points_rgb_np is not None and model_uses_rgb)
    model_arch_args_ns = argparse.Namespace(
        num_classes=args.num_classes, embed_dim=args.embed_dim, k_neighbors=args.k_neighbors,
        pt_hidden_dim=args.pt_hidden_dim, pt_heads=args.pt_heads,
        num_transformer_layers=args.num_transformer_layers, dropout=args.dropout,
        no_rgb=not current_sample_uses_rgb
    )
    try:
        model = PyG_PointTransformerSegModel(num_classes=model_arch_args_ns.num_classes, args=model_arch_args_ns).to(
            device)
        # For security, if you know the checkpoint only contains weights, use weights_only=True
        # However, to be safe with potentially complex checkpoints, weights_only=False is used here.
        checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(
            checkpoint_data['model_state_dict'] if 'model_state_dict' in checkpoint_data else checkpoint_data)
        model.eval();
        print("Semantic model loaded.")
    except Exception as e:
        print(f"FATAL Error loading semantic model: {e}");
        sys.exit(1)

    # --- 语义分割推理 ---
    print("\nPerforming semantic segmentation...")
    features_list = [points_original_np]
    if current_sample_uses_rgb:
        features_list.append(points_rgb_np.astype(np.float32) / 255.0)
    elif not model_arch_args_ns.no_rgb:  # Model expects RGB, but sample doesn't have it
        print("Warning: Model expects RGB, but sample has no RGB. Using dummy RGB.")
        features_list.append(np.full((points_original_np.shape[0], 3), 0.5, dtype=np.float32))

    features_np = np.concatenate(features_list, axis=1)
    features_tensor = torch.from_numpy(features_np).float().unsqueeze(0).to(device)
    expected_model_input_dim = 6 if not model_arch_args_ns.no_rgb else 3
    if features_tensor.shape[2] != expected_model_input_dim:
        print(f"FATAL: Feature tensor dim {features_tensor.shape[2]} != model expected dim {expected_model_input_dim}");
        sys.exit(1)

    with torch.no_grad():
        logits = model(features_tensor)
    pred_semantic_labels_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
    print("Semantic prediction complete.")
    unique_labels_viz, counts_viz = np.unique(pred_semantic_labels_np, return_counts=True)
    print(f"  DEBUG: Semantic prediction distribution: {dict(zip(unique_labels_viz, counts_viz))}")

    # --- DBSCAN 聚类 ---
    print("\nPerforming DBSCAN clustering...")
    target_points_xyz_semantic = points_original_np[pred_semantic_labels_np == args.target_label_id]
    instance_points_dict = {}

    print(f"  Points with target label {args.target_label_id} before DBSCAN: {target_points_xyz_semantic.shape[0]}")
    if args.visualize_intermediate_pcds and target_points_xyz_semantic.shape[0] > 0:
        target_label_viz_pcd = o3d.geometry.PointCloud()
        target_label_viz_pcd.points = o3d.utility.Vector3dVector(target_points_xyz_semantic)
        visualize_single_pcd(target_label_viz_pcd, window_name=f"Points for DBSCAN (Label {args.target_label_id})")

    if target_points_xyz_semantic.shape[0] >= args.dbscan_min_samples:
        try:
            db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
            instance_labels = db.fit_predict(target_points_xyz_semantic)
            unique_instances = sorted(np.unique(instance_labels[instance_labels != -1]))
            print(f"DBSCAN found {len(unique_instances)} instances with label {args.target_label_id}.")
            for inst_id_val in unique_instances:  # Renamed inst_id to inst_id_val to avoid conflict
                instance_points_dict[inst_id_val] = target_points_xyz_semantic[instance_labels == inst_id_val]
        except Exception as e:
            print(f"Error during DBSCAN: {e}")
            instance_points_dict = {}  # Reset on error
    else:
        print(
            f"Not enough target points for DBSCAN ({target_points_xyz_semantic.shape[0]} < {args.dbscan_min_samples}).")

    # --- 加载目标 CAD/扫描模型 ---
    print(f"\nLoading target model: {args.model_file}")
    if not os.path.exists(args.model_file): print(f"FATAL: Target model file not found: {args.model_file}"); sys.exit(1)

    # Load mesh for visualization, sample points for registration
    target_mesh_geometry_for_vis = o3d.io.read_triangle_mesh(args.model_file)
    if not target_mesh_geometry_for_vis.has_vertices():
        print("Warning: Could not load target as mesh, attempting to load as point cloud for visualization too.")
        target_mesh_geometry_for_vis = o3d.io.read_point_cloud(args.model_file)  # Fallback for vis
        if not target_mesh_geometry_for_vis.has_points():
            print("FATAL: Target model file (for vis) contains no points/vertices.");
            sys.exit(1)
        target_pcd_original_model = copy.deepcopy(target_mesh_geometry_for_vis)  # Use the pcd directly
        print(f"  Loaded target as point cloud with {len(target_pcd_original_model.points)} points.")
    else:
        target_pcd_original_model = target_mesh_geometry_for_vis.sample_points_uniformly(
            number_of_points=args.model_sample_points)
        print(f"  Loaded target as mesh, sampled {len(target_pcd_original_model.points)} points for registration.")

    if not target_pcd_original_model.has_points(): print(
        "FATAL: Sampled target model for registration has no points."); sys.exit(1)

    target_centroid_original_model = target_pcd_original_model.get_center()
    target_pcd_model_centered = copy.deepcopy(target_pcd_original_model).translate(-target_centroid_original_model)
    print(f"  Target model for registration centered. Original center: {target_centroid_original_model}")

    # --- 预计算目标模型的FPFH特征 (如果需要粗匹配) ---
    target_down_fpfh, target_fpfh_features = None, None
    if args.coarse_method != 'none':
        print("\nPreprocessing target model for FPFH-based coarse registration...")
        if not target_pcd_model_centered.has_points():
            print("ERROR: Centered target model (for FPFH) has no points. Coarse registration will be skipped.")
            args.coarse_method = 'none'
        else:
            target_down_fpfh, target_fpfh_features = preprocess_point_cloud_for_fpfh(
                target_pcd_model_centered, args.fpfh_voxel_size,
                args.fpfh_normal_radius_factor, args.fpfh_feature_radius_factor,
                args.fpfh_min_points, name="Target Model (Centered for FPFH)"
            )
            if target_down_fpfh is None or target_fpfh_features is None:
                print("ERROR: Failed to compute FPFH features for target model. Coarse registration will be skipped.")
                args.coarse_method = 'none'

    estimated_poses = {}
    if not instance_points_dict:
        print("\nNo instances found by DBSCAN, skipping pose estimation.")
    else:
        print(f"\n--- Starting Coarse-to-Fine Pose Estimation for {len(instance_points_dict)} identified instances ---")

    for inst_id, current_instance_points_np in instance_points_dict.items():
        print(f"\n--- Processing Instance ID: {inst_id} ({current_instance_points_np.shape[0]} points) ---")

        instance_pcd_for_registration = o3d.geometry.PointCloud()
        instance_pcd_for_registration.points = o3d.utility.Vector3dVector(current_instance_points_np)

        if len(instance_pcd_for_registration.points) < args.icp_min_points:
            print(
                f"  Skipping instance: Too few points ({len(instance_pcd_for_registration.points)} < {args.icp_min_points} required for ICP).")
            continue

        if args.visualize_intermediate_pcds:  # Visualize instance from DBSCAN
            visualize_single_pcd(instance_pcd_for_registration, window_name=f"Instance {inst_id} from DBSCAN (for Reg)")

        source_instance_centroid_original = instance_pcd_for_registration.get_center()
        source_instance_pcd_centered = copy.deepcopy(instance_pcd_for_registration).translate(
            -source_instance_centroid_original)

        initial_transform_for_fine_icp = np.identity(4)  # This is T_coarse_src_centered_to_tgt_centered

        # --- COARSE REGISTRATION ---
        if args.coarse_method != 'none':
            print(f"\n  Attempting Coarse Registration using: {args.coarse_method.upper()}")
            if not source_instance_pcd_centered.has_points():
                print("  ERROR: Centered source instance has no points. Skipping coarse registration.")
            elif target_down_fpfh is None or target_fpfh_features is None:  # Check if target FPFH was successful
                print("  ERROR: Target FPFH features not available. Skipping coarse registration for this instance.")
            else:
                print(f"    Preprocessing source instance {inst_id} for FPFH...")
                source_down_fpfh, source_fpfh_features = preprocess_point_cloud_for_fpfh(
                    source_instance_pcd_centered, args.fpfh_voxel_size,
                    args.fpfh_normal_radius_factor, args.fpfh_feature_radius_factor,
                    args.fpfh_min_points, name=f"Source Instance {inst_id} (Centered for FPFH)"
                )

                if source_down_fpfh and source_fpfh_features:
                    coarse_reg_result = None
                    coarse_time_start = time.time()
                    min_fitness_coarse = 0.0  # Default, will be overridden

                    if args.coarse_method == 'ransac':
                        distance_threshold_ransac = args.fpfh_voxel_size * args.ransac_distance_multiplier
                        print(f"    RANSAC: distance_threshold = {distance_threshold_ransac:.4f}")
                        coarse_reg_result = o3d_reg.registration_ransac_based_on_feature_matching(
                            source_down_fpfh, target_down_fpfh, source_fpfh_features, target_fpfh_features, True,
                            # mutual_filter
                            distance_threshold_ransac,
                            o3d_reg.TransformationEstimationPointToPoint(False), args.ransac_n_points,
                            [o3d_reg.CorrespondenceCheckerBasedOnEdgeLength(args.ransac_edge_length_factor),
                             o3d_reg.CorrespondenceCheckerBasedOnDistance(distance_threshold_ransac)],
                            o3d_reg.RANSACConvergenceCriteria(args.ransac_max_iter, args.ransac_confidence)
                        )
                        print("    RANSAC coarse_reg_result:", coarse_reg_result)
                        min_fitness_coarse = args.ransac_min_fitness

                    elif args.coarse_method == 'fgr':
                        fgr_options = o3d_reg.FastGlobalRegistrationOption(
                            maximum_correspondence_distance=args.fpfh_voxel_size * args.fgr_distance_multiplier,
                            iteration_number=args.fgr_iteration_number,
                            decrease_mu=args.fgr_decrease_mu,
                            tuple_scale=args.fgr_tuple_scale,
                            maximum_tuple_count=args.fgr_maximum_tuple_count
                        )
                        print(
                            f"    FGR: max_correspondence_distance = {fgr_options.maximum_correspondence_distance:.4f}")
                        coarse_reg_result = o3d_reg.registration_fgr_based_on_feature_matching(
                            source_down_fpfh, target_down_fpfh, source_fpfh_features, target_fpfh_features, fgr_options
                        )
                        print("    FGR coarse_reg_result:", coarse_reg_result)
                        min_fitness_coarse = args.fgr_min_fitness

                    coarse_time_end = time.time()
                    print(
                        f"    Coarse registration ({args.coarse_method.upper()}) took {coarse_time_end - coarse_time_start:.3f}s.")

                    if coarse_reg_result and coarse_reg_result.fitness >= min_fitness_coarse and np.any(
                            coarse_reg_result.transformation):  # Check if transformation is not None/empty
                        initial_transform_for_fine_icp = coarse_reg_result.transformation
                        print(
                            f"    Coarse Registration ({args.coarse_method.upper()}) successful. Fitness: {coarse_reg_result.fitness:.4f}, RMSE: {coarse_reg_result.inlier_rmse:.4f}")

                        if args.visualize_coarse_alignment:  # Visualizes alignment of FPFH inputs
                            source_coarsely_aligned_fpfh_input_vis = copy.deepcopy(source_down_fpfh).transform(
                                initial_transform_for_fine_icp)
                            visualize_intermediate_alignment(
                                source_coarsely_aligned_fpfh_input_vis, target_down_fpfh,
                                window_name=f"Coarse Align (FPFH inputs) ({args.coarse_method.upper()}) - Inst {inst_id}"
                            )

                        if args.visualize_coarse_pose_in_scene:  # Visualizes coarse pose on original model vs original instance
                            T_translate_to_Cs_orig_viz = np.eye(4);
                            T_translate_to_Cs_orig_viz[:3, 3] = source_instance_centroid_original
                            T_translate_from_Ct_orig_model_viz = np.eye(4);
                            T_translate_from_Ct_orig_model_viz[:3, 3] = -target_centroid_original_model
                            try:
                                pose_from_coarse_reg_viz = T_translate_to_Cs_orig_viz @ np.linalg.inv(
                                    initial_transform_for_fine_icp) @ T_translate_from_Ct_orig_model_viz
                                model_transformed_by_coarse_pose = copy.deepcopy(target_pcd_original_model).transform(
                                    pose_from_coarse_reg_viz)
                                visualize_pose_alignment_of_model_to_instance(
                                    model_transformed_by_coarse_pose,
                                    instance_pcd_for_registration,  # Original instance from DBSCAN
                                    window_name=f"Coarse Pose in Scene ({args.coarse_method.upper()}) - Inst {inst_id}",
                                    model_name_in_legend="CAD Model transformed by COARSE pose"
                                )
                            except np.linalg.LinAlgError:
                                print("    Error: Singular matrix in visualizing coarse pose in scene.")


                    else:
                        fitness_val = coarse_reg_result.fitness if coarse_reg_result else -1.0
                        print(
                            f"    Coarse Registration ({args.coarse_method.upper()}) failed or fitness too low ({fitness_val:.4f} < {min_fitness_coarse}). Using identity for ICP.")
                else:
                    print(
                        "    Skipping coarse registration for this instance due to FPFH feature computation failure for source.")
        else:  # coarse_method == 'none'
            print("\n  Skipping Coarse Registration (method: none). Using identity for ICP.")

        # --- FINE REGISTRATION (ICP) ---
        print("\n  Performing Fine ICP Registration...")
        fine_icp_estimation_method = None
        if args.icp_estimation_method.lower() == 'point_to_point':
            fine_icp_estimation_method = o3d_reg.TransformationEstimationPointToPoint()
        elif args.icp_estimation_method.lower() == 'point_to_plane':
            normal_radius_icp = args.icp_threshold * 2.0
            # print(f"    Estimating normals for Point-to-Plane ICP (radius: {normal_radius_icp:.4f})...") # Verbose
            source_instance_pcd_centered.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_icp, max_nn=30))
            if not target_pcd_model_centered.has_normals():
                target_pcd_model_centered.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_icp, max_nn=30))

            if source_instance_pcd_centered.has_normals() and target_pcd_model_centered.has_normals():
                fine_icp_estimation_method = o3d_reg.TransformationEstimationPointToPlane()
            else:
                print("    Normal estimation failed for Point-to-Plane ICP. Switching to Point-to-Point.")
                fine_icp_estimation_method = o3d_reg.TransformationEstimationPointToPoint()
                if source_instance_pcd_centered.has_normals(): source_instance_pcd_centered.normals.clear()

        criteria_fine_icp = o3d_reg.ICPConvergenceCriteria(
            relative_fitness=args.icp_relative_fitness, relative_rmse=args.icp_relative_rmse,
            max_iteration=args.icp_max_iter
        )

        fine_icp_start_time = time.time()
        fine_reg_result = o3d_reg.registration_icp(
            source_instance_pcd_centered, target_pcd_model_centered,  # Use full centered clouds for ICP
            args.icp_threshold, initial_transform_for_fine_icp,
            # initial_transform is T_coarse_src_centered_to_tgt_centered
            fine_icp_estimation_method, criteria_fine_icp
        )
        fine_icp_end_time = time.time()
        print(f"    Fine ICP finished in {fine_icp_end_time - fine_icp_start_time:.3f}s.")
        print(f"    Fine ICP Result: Fitness={fine_reg_result.fitness:.4f}, RMSE={fine_reg_result.inlier_rmse:.4f}")

        T_fine_s_centered_to_t_centered = fine_reg_result.transformation
        T_translate_to_Cs_orig = np.eye(4);
        T_translate_to_Cs_orig[:3, 3] = source_instance_centroid_original
        T_translate_from_Ct_orig_model = np.eye(4);
        T_translate_from_Ct_orig_model[:3, 3] = -target_centroid_original_model

        try:
            # Final pose transforms original model to align with original instance in scene
            estimated_pose = T_translate_to_Cs_orig @ np.linalg.inv(
                T_fine_s_centered_to_t_centered) @ T_translate_from_Ct_orig_model

            # print("\n  Calculated Final Pose Matrix:") # Verbose
            # print(estimated_pose)
            estimated_poses[inst_id] = estimated_pose

            if args.visualize_fine_alignment:
                transformed_model_pcd_for_final_align = copy.deepcopy(target_pcd_original_model).transform(
                    estimated_pose)
                visualize_pose_alignment_of_model_to_instance(
                    transformed_model_pcd_for_final_align,
                    instance_pcd_for_registration,  # Original instance from DBSCAN
                    window_name=f"Fine ICP Alignment - Inst {inst_id}",
                    model_name_in_legend="CAD Model transformed by FINAL pose"
                )

            if args.visualize_pose_in_scene:
                # Use the original mesh for visualization if available, otherwise the original sampled point cloud
                model_geometry_to_show_in_scene = target_mesh_geometry_for_vis \
                    if isinstance(target_mesh_geometry_for_vis,
                                  o3d.geometry.TriangleMesh) and target_mesh_geometry_for_vis.has_vertices() \
                    else target_pcd_original_model

                visualize_transformed_model_in_scene(
                    original_scene_o3d_pcd, model_geometry_to_show_in_scene, estimated_pose,
                    window_name=f"Final Model in Full Scene - Instance {inst_id}"
                )
        except np.linalg.LinAlgError:
            print(
                f"  Error: Singular matrix encountered during final pose calculation for instance {inst_id}. Skipping this instance.")
        except Exception as e_pose:
            print(f"  Error during final pose processing for instance {inst_id}: {e_pose}")

    if args.save_results and estimated_poses:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        pose_filename = os.path.join(args.output_dir, f"estimated_poses_sample_{args.sample_index}_{timestamp}.npz")
        try:
            np.savez(pose_filename, **{f'instance_{k}': v for k, v in estimated_poses.items()})
            print(f"\nSaved estimated poses for {len(estimated_poses)} instances to {pose_filename}")
        except Exception as e_save:
            print(f"\nError saving poses: {e_save}")

    print("\n--- Pose estimation script finished. ---")


if __name__ == "__main__":
    # Argument parsing is kept the same as provided by the user in the prompt
    parser = argparse.ArgumentParser(
        description='在HDF5样本上执行实例分割和粗到精姿态估计 (单位: 毫米)。')

    io_group = parser.add_argument_group('输入/输出 (Input/Output)')
    io_group.add_argument('--input_h5', type=str, default='./data/my_custom_dataset_h5_rgb/test_0.h5',
                          help='输入的HDF5文件路径。')
    io_group.add_argument('--sample_index', type=int, default=1, help='HDF5文件中的样本索引。')
    io_group.add_argument('--checkpoint', type=str, default="checkpoints_seg_pyg_ptconv_rgb/best_model.pth",
                          help='语义分割模型的检查点文件路径。')
    io_group.add_argument('--model_file', type=str, default="stp/cube.STL",
                          help='目标三维模型文件路径 (例如：STL, PLY, OBJ)，单位需与场景点云一致 (毫米)。')
    io_group.add_argument('--output_dir', type=str, default='./pose_estimation_results_cmdline',
                          help='保存结果的目录路径。')

    data_proc_group = parser.add_argument_group('数据处理 (Data Processing)')
    data_proc_group.add_argument('--no_rgb', action='store_true', help="不使用HDF5文件中的RGB数据（即使存在）。")
    data_proc_group.add_argument('--model_sample_points', type=int, default=2048*2,
                                 help='从CAD模型上采样用于配准的点数量 (单位: 毫米时，点数应能覆盖模型表面)。')

    dbscan_group = parser.add_argument_group('语义目标与DBSCAN聚类 (Semantic Target & DBSCAN - Units: mm)')
    dbscan_group.add_argument('--target_label_id', type=int, default=1, help='用于DBSCAN聚类的目标语义标签ID。')
    dbscan_group.add_argument('--dbscan_eps', type=float, default=200.0,
                              help='DBSCAN的epsilon邻域距离 (单位: 毫米)。例如，对于点间距几毫米的物体表面，可设为10-20mm。')
    dbscan_group.add_argument('--dbscan_min_samples', type=int, default=1000,
                              help='DBSCAN形成核心点的最小样本数。例如，对于一个小的实例块，50-100个点可能比较合适。')

    coarse_method_group = parser.add_argument_group('粗匹配方法 (Coarse Registration Method)')
    coarse_method_group.add_argument('--coarse_method', type=str, default='ransac', choices=['none', 'ransac', 'fgr'],
                                     help="选择粗匹配的方法。'none'表示跳过粗匹配阶段。'ransac'使用FPFH+RANSAC，'fgr'使用Fast Global Registration。")

    fpfh_group = parser.add_argument_group(
        'FPFH特征参数 (用于粗匹配 - Units: mm) (FPFH Parameters for Coarse Registration)')
    fpfh_group.add_argument('--fpfh_voxel_size', type=float, default=5.0,
                            help='FPFH特征计算前进行体素下采样的体素大小 (单位: 毫米)。例如，对于50-150mm的物体，可设为2.5mm, 5mm或7.5mm。')
    fpfh_group.add_argument('--fpfh_normal_radius_factor', type=float, default=2.0,
                            help='计算FPFH法线时，搜索半径为 fpfh_voxel_size 乘以此因子。(例如 5mm * 2.0 = 10mm)')
    fpfh_group.add_argument('--fpfh_feature_radius_factor', type=float, default=1.0,
                            help='计算FPFH特征时，搜索半径为 fpfh_voxel_size 乘以此因子。 (例如 5mm * 5.0 = 25mm)')
    fpfh_group.add_argument('--fpfh_min_points', type=int, default=20,
                            help='点云在体素下采样后，尝试计算FPFH特征所需的最少点数。')

    ransac_coarse_group = parser.add_argument_group(
        'RANSAC粗匹配参数 (如果 coarse_method 为 ransac) (RANSAC Coarse Registration Parameters - Units: mm)')
    ransac_coarse_group.add_argument('--ransac_distance_multiplier', type=float, default=1.5,
                                     help='RANSAC对应点距离阈值为 fpfh_voxel_size 乘以此因子。(例如 5mm * 1.5 = 7.5mm)')
    ransac_coarse_group.add_argument('--ransac_edge_length_factor', type=float, default=0.9,
                                     help='RANSAC对应关系检查器：边长因子。')
    ransac_coarse_group.add_argument('--ransac_n_points', type=int, default=3,
                                     help='RANSAC每次迭代采样的点对数量。')
    ransac_coarse_group.add_argument('--ransac_max_iter', type=int, default=100000, help='RANSAC的最大迭代次数。')
    ransac_coarse_group.add_argument('--ransac_confidence', type=float, default=0.999,
                                     help='RANSAC置信度 (目标概率)。')
    ransac_coarse_group.add_argument('--ransac_min_fitness', type=float, default=0.1,
                                     help='认为RANSAC粗匹配成功的最小适应度 (fitness)。')

    fgr_coarse_group = parser.add_argument_group(
        'FGR粗匹配参数 (如果 coarse_method 为 fgr) (FGR Coarse Registration Parameters - Units: mm)')
    fgr_coarse_group.add_argument('--fgr_distance_multiplier', type=float, default=1.5,
                                  help='FGR最大对应距离为 fpfh_voxel_size 乘以此因子。(例如 5mm * 1.5 = 7.5mm)')
    fgr_coarse_group.add_argument('--fgr_iteration_number', type=int, default=64, help='FGR的迭代次数。')
    fgr_coarse_group.add_argument('--fgr_decrease_mu', type=float, default=0.97,
                                  help='FGR的mu衰减系数 (Open3D默认值为0.97)。')
    fgr_coarse_group.add_argument('--fgr_tuple_scale', type=float, default=0.95, help='FGR的元组尺度。')
    fgr_coarse_group.add_argument('--fgr_maximum_tuple_count', type=int, default=1000, help='FGR的最大元组数。')
    fgr_coarse_group.add_argument('--fgr_min_fitness', type=float, default=0.05,
                                  help='认为FGR粗匹配成功的最小适应度 (fitness)。')

    icp_group = parser.add_argument_group('精细ICP参数 (Fine ICP Parameters - Units: mm)')
    icp_group.add_argument('--icp_threshold', type=float, default=2.0,
                           help='精细ICP的最大对应点距离阈值 (单位: 毫米)。例如，在粗对齐后，期望误差在1-5mm内，可设为2mm或5mm。')
    icp_group.add_argument('--icp_estimation_method', type=str, default='point_to_point',
                           choices=['point_to_point', 'point_to_plane'],
                           help="精细ICP的变换估计算法 ('点对点' 或 '点对面')。")
    icp_group.add_argument('--icp_relative_fitness', type=float, default=1e-06,
                           help='精细ICP收敛条件：相对适应度变化阈值。')
    icp_group.add_argument('--icp_relative_rmse', type=float, default=1e-06,
                           help='精细ICP收敛条件：相对均方根误差变化阈值。')
    icp_group.add_argument('--icp_max_iter', type=int, default=1000, help='精细ICP收敛条件：最大迭代次数。')
    icp_group.add_argument('--icp_min_points', type=int, default=30,
                           help='实例点云在DBSCAN之后，进行任何配准尝试所需的最少点数。')

    ctrl_group = parser.add_argument_group('控制与可视化 (Control & Visualization)')
    ctrl_group.add_argument('--no_cuda', action='store_true', help='禁用CUDA，使用CPU进行计算。')
    ctrl_group.add_argument('--save_results', action='store_true', help='将估计的姿态保存到 .npz 文件。')
    ctrl_group.add_argument('--visualize_intermediate_pcds', action='store_true', default=True,
                            help='可视化DBSCAN聚类前(带目标标签)和后(实例)的点云。')
    ctrl_group.add_argument('--visualize_coarse_alignment', action='store_true', default=True,
                            help='可视化粗匹配中FPFH输入点云的对齐结果。')
    # 新增参数用于可视化粗匹配姿态作用于原始模型和原始实例的效果
    ctrl_group.add_argument('--visualize_coarse_pose_in_scene', action='store_true', default=True,
                            help='可视化粗匹配得到的姿态应用于原始模型和原始实例的对齐结果。')
    ctrl_group.add_argument('--visualize_fine_alignment', action='store_true', default=True,
                            help='可视化精细ICP的对齐结果 (变换后的模型 vs 观察到的实例)。')
    ctrl_group.add_argument('--visualize_pose_in_scene', action='store_true', default=True,
                            help='可视化最终变换后的模型 (STL/CAD) 在原始完整场景点云中的位置。')

    model_arch_group = parser.add_argument_group(
        '模型架构 (需与训练配置匹配) (Model Architecture - match training config)')
    model_arch_group.add_argument('--num_classes', type=int, default=2, help='模型训练的语义类别数量。')
    model_arch_group.add_argument('--embed_dim', type=int, default=64, help='Point Transformer的初始嵌入维度。')
    model_arch_group.add_argument('--k_neighbors', type=int, default=16, help='Point Transformer中k-NN图的k值。')
    model_arch_group.add_argument('--pt_hidden_dim', type=int, default=128, help='Point Transformer的隐藏层维度。')
    model_arch_group.add_argument('--pt_heads', type=int, default=4, help='Point Transformer的注意力头数量。')
    model_arch_group.add_argument('--num_transformer_layers', type=int, default=2, help='Point Transformer的层数。')
    model_arch_group.add_argument('--dropout', type=float, default=0.3, help='模型训练时使用的dropout率。')

    args = parser.parse_args()
    main(args)

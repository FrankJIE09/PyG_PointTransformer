# estimate_pose_from_h5_cmdline.py
# 加载 HDF5 样本，执行语义分割+聚类，然后对每个实例进行 ICP 姿态估计。
# 所有 ICP 参数均通过命令行参数进行配置。不使用图形界面。

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
    sys.exit(1)

# --- 导入 Open3D and Scikit-learn ---
try:
    import open3d as o3d
    from open3d.pipelines import registration as o3d_reg

    OPEN3D_AVAILABLE = True
except ImportError:
    print("FATAL Error: Open3D not found. Please install Open3D (pip install open3d).")
    OPEN3D_AVAILABLE = False
try:
    from sklearn.cluster import DBSCAN

    SKLEARN_AVAILABLE = True
except ImportError:
    print("FATAL Error: scikit-learn not found. Please install scikit-learn (pip install scikit-learn).")
    SKLEARN_AVAILABLE = False


# --- 新的可视化函数: 显示单个点云 ---
def visualize_single_pcd(pcd, window_name="Point Cloud", point_color=[0.5, 0.5, 0.5]):
    """可视化单个点云。"""
    if not OPEN3D_AVAILABLE:
        print("Open3D not available, skipping visualization.")
        return
    if not pcd.has_points():
        print(f"Skipping visualization for {window_name}: No points to display.")
        return

    pcd_vis = copy.deepcopy(pcd)
    if not pcd_vis.has_colors():  # 如果点云没有颜色，则赋予默认颜色
        pcd_vis.paint_uniform_color(point_color)

    print(f"\nDisplaying Point Cloud: {window_name}")
    print(f"  Points: {len(pcd_vis.points)}")
    print("(Close the window to continue...)")
    try:
        o3d.visualization.draw_geometries([pcd_vis], window_name=window_name)
        print(f"Visualization window '{window_name}' closed.")
    except Exception as e:
        print(f"Error during visualization of '{window_name}': {e}")


# --- 可视化函数: 显示 ICP 对齐结果 (实例点 vs 模型点) ---
def visualize_icp_alignment(source_pcd_transformed_model, instance_pcd_observed, window_name="ICP Alignment Result"):
    """可视化对齐后的源点云（实例）和目标点云（模型）。
       source_pcd_transformed_model: 按照估计姿态变换后的目标模型点云 (通常是采样点)。
       instance_pcd_observed: 原始（可能经过预处理）的实例点云 (场景中的观察)。
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available, skipping visualization.")
        return

    instance_pcd_vis = copy.deepcopy(instance_pcd_observed)
    transformed_model_vis = copy.deepcopy(source_pcd_transformed_model)

    transformed_model_vis.paint_uniform_color([1, 0.706, 0])  # 黄色 (变换后的模型点)
    instance_pcd_vis.paint_uniform_color([0, 0.651, 0.929])  # 蓝色 (观察到的实例点)

    print(f"\nDisplaying ICP Alignment: {window_name}...")
    print("Yellow: Transformed Model Sampled Points | Blue: Observed Instance Points (preprocessed)")
    print("(Close the window to continue...)")
    try:
        o3d.visualization.draw_geometries([transformed_model_vis, instance_pcd_vis], window_name=window_name)
        print("Alignment visualization window closed.")
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Visualization window might not have displayed correctly.")


# --- 新的可视化函数: 显示变换后的模型在原始场景中 ---
def visualize_transformed_model_in_scene(original_scene_pcd, target_model_geometry, estimated_pose,
                                         window_name="Transformed Model in Original Scene"):
    """可视化变换后的目标模型（STL/CAD）在完整的原始场景点云中的位置。"""
    if not OPEN3D_AVAILABLE:
        print("Open3D not available, skipping visualization.")
        return

    scene_vis = copy.deepcopy(original_scene_pcd)
    model_vis = copy.deepcopy(target_model_geometry)  # This could be a mesh or a point cloud

    # Apply the pose to the model
    model_vis.transform(estimated_pose)

    # Assign colors
    # Scene cloud might already have colors from HDF5; if not, paint gray
    if not scene_vis.has_colors():
        scene_vis.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray for original scene

    if isinstance(model_vis, o3d.geometry.TriangleMesh):
        model_vis.paint_uniform_color([0, 1, 0])  # Green for transformed model mesh
        if not model_vis.has_vertex_normals():  # Ensure mesh has normals for better rendering
            model_vis.compute_vertex_normals()
    elif isinstance(model_vis, o3d.geometry.PointCloud):
        model_vis.paint_uniform_color([0, 1, 0])  # Green for transformed model pcd (if it was loaded as pcd)

    print(f"\nDisplaying: {window_name}")
    print(
        "Scene Color (e.g., Gray/Original RGB): Original Full Scene Point Cloud | Green: Transformed Target Model (STL/CAD)")
    print("(Close the window to continue...)")
    try:
        geometries_to_draw = [scene_vis]
        if isinstance(model_vis, o3d.geometry.TriangleMesh) and not model_vis.has_vertices():
            print("Warning: Transformed model mesh has no vertices, not drawing it.")
        elif isinstance(model_vis, o3d.geometry.PointCloud) and not model_vis.has_points():
            print("Warning: Transformed model point cloud has no points, not drawing it.")
        else:
            geometries_to_draw.append(model_vis)

        if len(geometries_to_draw) > 1:
            o3d.visualization.draw_geometries(geometries_to_draw, window_name=window_name)
            print(f"Visualization window '{window_name}' closed.")
        else:
            print(f"Not enough geometries to display for '{window_name}'.")

    except Exception as e:
        print(f"Error during visualization of '{window_name}': {e}")


# --- 主函数 ---
def main(args):
    print(f"Starting Pose Estimation from HDF5 at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    if not SKLEARN_AVAILABLE:
        print("FATAL: scikit-learn required.")
        sys.exit(1)
    if not OPEN3D_AVAILABLE:
        print("FATAL: Open3D required.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- 加载和预处理 HDF5 数据 ---
    print(f"Loading data from HDF5: {args.input_h5}, Sample: {args.sample_index}")
    points_original_np = None
    points_rgb_np = None
    model_uses_rgb = False

    if not os.path.exists(args.input_h5):
        print(f"FATAL Error: HDF5 file not found: {args.input_h5}")
        sys.exit(1)

    try:
        with h5py.File(args.input_h5, 'r') as f:
            if 'data' not in f: raise KeyError("'data' key missing in HDF5 file.")
            dset_data = f['data']
            dset_rgb = f.get('rgb', None)

            if not (0 <= args.sample_index < dset_data.shape[0]):
                raise IndexError(f"Sample index {args.sample_index} out of bounds (0 to {dset_data.shape[0] - 1}).")

            points_original_np = dset_data[args.sample_index].astype(np.float32)

            if dset_rgb is not None and not args.no_rgb:
                if dset_rgb.shape[0] > args.sample_index and dset_rgb[args.sample_index].shape[0] == \
                        points_original_np.shape[0]:
                    points_rgb_np = dset_rgb[args.sample_index].astype(np.uint8)
                    model_uses_rgb = True
                    print("RGB data found in HDF5 and enabled.")
                else:
                    print(
                        f"Warning: RGB data in HDF5 has issues for sample index {args.sample_index} or mismatched points. Using XYZ only.")
                    points_rgb_np = None  # Ensure it's None
                    model_uses_rgb = False
            else:
                print("RGB data not found in HDF5 or disabled via --no_rgb flag. Using XYZ only.")
                model_uses_rgb = False
    except Exception as e:
        print(f"FATAL Error loading/preprocessing HDF5: {e}")
        sys.exit(1)

    if points_original_np is None:
        print("FATAL Error: Points data could not be loaded.")
        sys.exit(1)

    # Create Open3D point cloud for the original full scene
    original_scene_o3d_pcd = o3d.geometry.PointCloud()
    original_scene_o3d_pcd.points = o3d.utility.Vector3dVector(points_original_np)
    if points_rgb_np is not None and model_uses_rgb:  # model_uses_rgb implies not args.no_rgb
        if points_rgb_np.shape[0] == points_original_np.shape[0]:
            rgb_normalized = points_rgb_np.astype(np.float64) / 255.0
            original_scene_o3d_pcd.colors = o3d.utility.Vector3dVector(rgb_normalized)
            print("Applied RGB colors to the original scene point cloud for visualization.")
        else:
            print(
                "Warning: Mismatch between points and RGB for original scene visualization. Scene will not be colored with RGB.")
    else:
        print(
            "Original scene point cloud will be visualized without RGB (will use default color in viz function if needed).")

    # --- 加载语义模型 ---
    print(f"\nLoading SEMANTIC model checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"FATAL Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Determine model input channels based on actual RGB availability for this sample
    current_sample_uses_rgb = (points_rgb_np is not None and model_uses_rgb)
    model_input_channels = 6 if current_sample_uses_rgb else 3
    print(f"Initializing model for {model_input_channels}D input (RGB used: {current_sample_uses_rgb})...")

    model_args = argparse.Namespace(
        num_classes=args.num_classes, embed_dim=args.embed_dim, k_neighbors=args.k_neighbors,
        pt_hidden_dim=args.pt_hidden_dim, pt_heads=args.pt_heads,
        num_transformer_layers=args.num_transformer_layers, dropout=args.dropout,
        no_rgb=not current_sample_uses_rgb  # Pass the actual RGB usage for the sample to model
    )
    try:
        model = PyG_PointTransformerSegModel(num_classes=model_args.num_classes, args=model_args).to(device)
        print("Model structure initialized.")
        checkpoint_data = torch.load(args.checkpoint, map_location=device,
                                     weights_only=False)  # weights_only=False if optimizer state etc. might be there
        if 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        else:
            model.load_state_dict(checkpoint_data)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"FATAL Error initializing or loading model: {e}")
        sys.exit(1)
    model.eval()

    # --- 语义分割推理 ---
    print("\nPerforming semantic segmentation inference...")
    features_list = [points_original_np]
    if current_sample_uses_rgb:  # Use the flag reflecting actual data for this sample
        features_list.append(points_rgb_np.astype(np.float32) / 255.0)
        print("Using XYZ+RGB features for inference.")
    else:
        # If model was trained for RGB but this sample has no RGB, we need to provide dummy RGB
        if not model_args.no_rgb and not current_sample_uses_rgb:  # Model expects RGB, but current sample doesn't have it
            print(
                "Warning: Model expects RGB but current sample has no RGB. Using XYZ + dummy RGB features for inference.")
            features_list.append(np.full((points_original_np.shape[0], 3), 0.5, dtype=np.float32))
        else:  # Model expects XYZ, or model expects RGB and current sample has it (covered above)
            print("Using XYZ features for inference.")

    features_np = np.concatenate(features_list, axis=1)
    features_tensor = torch.from_numpy(features_np).float().unsqueeze(0).to(device)
    expected_feature_dim = 6 if not model_args.no_rgb else 3  # Expected by model architecture

    if features_np.shape[1] != expected_feature_dim:
        print(
            f"FATAL Error: Prepared features dimension ({features_np.shape[1]}D) does not match model expected dimension ({expected_feature_dim}D based on model's no_rgb={model_args.no_rgb}).")
        sys.exit(1)

    with torch.no_grad():
        logits = model(features_tensor)
    pred_semantic_labels_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
    print("Semantic prediction complete.")
    unique_semantic, counts_semantic = np.unique(pred_semantic_labels_np, return_counts=True)
    print(f"  Predicted semantic label distribution: {dict(zip(unique_semantic, counts_semantic))}")

    # --- DBSCAN 聚类 ---
    print("\nPerforming DBSCAN clustering...")
    target_label_id = args.target_label_id
    target_mask = (pred_semantic_labels_np == target_label_id)
    target_points_xyz = points_original_np[target_mask]

    num_instances_found = 0
    instance_points_dict = {}

    if target_points_xyz.shape[0] >= args.dbscan_min_samples:
        try:
            db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
            instance_labels_target_points = db.fit_predict(target_points_xyz)
            unique_instances = np.unique(
                instance_labels_target_points[instance_labels_target_points != -1])  # Exclude noise label -1
            unique_instances.sort()  # Ensure consistent order if needed later
            num_instances_found = len(unique_instances)
            print(f"Clustering found {num_instances_found} potential instances with label {target_label_id}.")
            for inst_id in unique_instances:
                instance_mask_local = (instance_labels_target_points == inst_id)
                instance_points_dict[inst_id] = target_points_xyz[instance_mask_local]
                print(f"  Instance {inst_id}: {instance_points_dict[inst_id].shape[0]} points")
        except Exception as e:
            print(f"Error during DBSCAN clustering: {e}")
            num_instances_found = 0  # Reset on error
            instance_points_dict = {}  # Reset on error
    else:
        print(
            f"Not enough points predicted as target (label {target_label_id}) for DBSCAN (required: {args.dbscan_min_samples}, found: {target_points_xyz.shape[0]}).")

    # --- 加载目标 CAD/扫描模型 ---
    print(f"\nLoading target model file from: {args.model_file}")
    target_mesh = o3d.geometry.TriangleMesh()  # Initialize an empty mesh
    target_pcd_original_model = o3d.geometry.PointCloud()  # Initialize an empty point cloud
    target_centroid_original_model = np.zeros(3)

    if not os.path.exists(args.model_file):
        print(f"FATAL Error: Target model file not found: {args.model_file}")
        sys.exit(1)
    try:
        # Attempt to read as mesh first
        temp_mesh = o3d.io.read_triangle_mesh(args.model_file)
        if temp_mesh.has_vertices():
            target_mesh = temp_mesh  # Store the loaded mesh
            print(f"  Loaded mesh with {len(target_mesh.vertices)} vertices, {len(target_mesh.triangles)} triangles.")
            # Sample points from this mesh for ICP if needed, or use the mesh directly for visualization
            num_model_points_to_sample = args.model_sample_points # Heuristic for sampling
            target_pcd_original_model = target_mesh.sample_points_uniformly(number_of_points=num_model_points_to_sample)
            if not target_pcd_original_model.has_points():
                raise ValueError("Sampling points from loaded mesh failed or resulted in zero points.")
            print(f"  Sampled {len(target_pcd_original_model.points)} points from mesh for ICP.")
        else:
            print("  Could not load as mesh or mesh is empty, attempting to load as point cloud...")
            target_pcd_original_model = o3d.io.read_point_cloud(args.model_file)
            if not target_pcd_original_model.has_points():
                raise ValueError("Target model file contains no points when loaded as PointCloud.")
            print(f"  Loaded target model as point cloud with {len(target_pcd_original_model.points)} points.")
            # target_mesh remains empty if loaded as point cloud

        # Common processing for target_pcd_original_model (whether sampled or loaded directly)
        target_centroid_original_model = target_pcd_original_model.get_center()
        print(f"Target model (used for ICP) has {len(target_pcd_original_model.points)} points.")
        print(
            f"  Original Target Model (PCD) Center: [{target_centroid_original_model[0]:.3f},{target_centroid_original_model[1]:.3f},{target_centroid_original_model[2]:.3f}]")

        target_pcd_model_centered = copy.deepcopy(target_pcd_original_model)
        target_pcd_model_centered.translate(-target_centroid_original_model)  # Center the PCD version for ICP
        print(f"  Target model (PCD) centered around origin for ICP.")

    except Exception as e:
        print(f"FATAL Error loading or processing target model file: {e}")
        sys.exit(1)

    # --- Configure ICP Parameters ---
    print("\nConfiguring ICP parameters from command-line arguments:")
    threshold = args.icp_threshold
    method_str = args.icp_estimation_method
    relative_fitness = args.icp_relative_fitness
    relative_rmse = args.icp_relative_rmse
    max_iteration = args.icp_max_iter
    min_points_icp = args.icp_min_points
    print(f"  ICP max_distance (threshold): {threshold}")
    print(f"  ICP estimation_method: {method_str}")
    print(
        f"  ICP convergence_criteria: relative_fitness={relative_fitness}, relative_rmse={relative_rmse}, max_iteration={max_iteration}")
    print(f"  ICP min_points: {min_points_icp}")

    # --- 对每个找到的实例执行 ICP 姿态估计 ---
    estimated_poses = {}
    if num_instances_found > 0:
        print("\nPerforming ICP for each found instance...")

        for inst_id, instance_points_np in instance_points_dict.items():
            print(f"\nProcessing Instance ID: {inst_id} (Initial points: {instance_points_np.shape[0]})")

            raw_instance_pcd = o3d.geometry.PointCloud()
            raw_instance_pcd.points = o3d.utility.Vector3dVector(instance_points_np)

            # --- BEGIN: Instance Preprocessing ---
            instance_pcd_for_icp = copy.deepcopy(raw_instance_pcd)
            print("  Preprocessing instance point cloud...")

            if args.preprocess_voxel_size > 0 and instance_pcd_for_icp.has_points():
                # (Preprocessing steps as before)
                print(f"    Applying Voxel Downsampling with voxel_size={args.preprocess_voxel_size}")
                instance_pcd_for_icp = instance_pcd_for_icp.voxel_down_sample(args.preprocess_voxel_size)
                print(f"    Points after Voxel Downsampling: {len(instance_pcd_for_icp.points)}")
                if not instance_pcd_for_icp.has_points():
                    print(f"  Skipping ICP for Instance {inst_id}: No points left after Voxel Downsampling.")
                    if args.visualize_intermediate_pcds: visualize_single_pcd(instance_pcd_for_icp,
                                                                              window_name=f"Instance {inst_id} After Voxel (Empty)",
                                                                              point_color=[1, 0, 0])
                    continue
            if args.preprocess_sor_k > 0 and args.preprocess_sor_std_ratio > 0 and instance_pcd_for_icp.has_points():
                print(
                    f"    Applying Statistical Outlier Removal (SOR) with k={args.preprocess_sor_k}, std_ratio={args.preprocess_sor_std_ratio}")
                sor_pcd, ind = instance_pcd_for_icp.remove_statistical_outlier(nb_neighbors=args.preprocess_sor_k,
                                                                               std_ratio=args.preprocess_sor_std_ratio)
                if sor_pcd.has_points() and len(ind) > 0:
                    instance_pcd_for_icp = sor_pcd
                else:
                    print("    Warning: SOR removed all points or resulted in an empty cloud. Using points before SOR.")
                print(f"    Points after SOR: {len(instance_pcd_for_icp.points)}")
                if not instance_pcd_for_icp.has_points():
                    print(f"  Skipping ICP for Instance {inst_id}: No points left after SOR.")
                    if args.visualize_intermediate_pcds: visualize_single_pcd(instance_pcd_for_icp,
                                                                              window_name=f"Instance {inst_id} After SOR (Empty)",
                                                                              point_color=[1, 0, 0])
                    continue
            if args.preprocess_fps_n_points > 0 and instance_pcd_for_icp.has_points():
                num_current_points = len(instance_pcd_for_icp.points)
                if num_current_points > args.preprocess_fps_n_points:
                    print(
                        f"    Applying Farthest Point Sampling to {args.preprocess_fps_n_points} points (was {num_current_points})")
                    instance_pcd_for_icp = instance_pcd_for_icp.farthest_point_down_sample(args.preprocess_fps_n_points)
                    print(f"    Points after FPS: {len(instance_pcd_for_icp.points)}")
                else:
                    print(
                        f"    Skipping FPS: Current points ({num_current_points}) <= target FPS points ({args.preprocess_fps_n_points}).")
                if not instance_pcd_for_icp.has_points():
                    print(f"  Skipping ICP for Instance {inst_id}: No points left after FPS.")
                    if args.visualize_intermediate_pcds: visualize_single_pcd(instance_pcd_for_icp,
                                                                              window_name=f"Instance {inst_id} After FPS (Empty)",
                                                                              point_color=[1, 0, 0])
                    continue
            # --- END: Instance Preprocessing ---

            if args.visualize_intermediate_pcds and instance_pcd_for_icp.has_points():
                visualize_single_pcd(instance_pcd_for_icp,
                                     window_name=f"Preprocessed Instance {inst_id} ({len(instance_pcd_for_icp.points)} pts)",
                                     point_color=[0.2, 0.8, 0.2])

            if len(instance_pcd_for_icp.points) < min_points_icp:
                print(
                    f"  Skipping ICP: Instance has only {len(instance_pcd_for_icp.points)} points after ALL preprocessing (min required: {min_points_icp}).")
                continue

            source_instance_centroid_original = instance_pcd_for_icp.get_center()
            print(
                f"  Preprocessed Instance Center (Cs): [{source_instance_centroid_original[0]:.3f},{source_instance_centroid_original[1]:.3f},{source_instance_centroid_original[2]:.3f}]")
            source_instance_pcd_centered = copy.deepcopy(instance_pcd_for_icp)
            source_instance_pcd_centered.translate(-source_instance_centroid_original)
            print(f"  Preprocessed instance centered for ICP.")

            estimation_method = None
            if method_str.lower() == 'point_to_point':
                estimation_method = o3d_reg.TransformationEstimationPointToPoint()
                print("  Using Point-to-Point ICP.")
            elif method_str.lower() == 'point_to_plane':
                print("  Using Point-to-Plane ICP.")
                print("    Estimating normals for source (instance) and target (model pcd)...")
                normal_radius = threshold * 2.0
                normal_radius = max(1e-3, normal_radius)  # Ensure positive radius
                try:
                    source_instance_pcd_centered.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
                    if not target_pcd_model_centered.has_normals():  # target_pcd_model_centered is used for ICP
                        target_pcd_model_centered.estimate_normals(
                            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

                    if source_instance_pcd_centered.has_normals() and target_pcd_model_centered.has_normals():
                        estimation_method = o3d_reg.TransformationEstimationPointToPlane()
                        print("    Normals estimated successfully for Point-to-Plane.")
                    else:
                        print(
                            "    Normal estimation failed or resulted in no normals for one or both clouds. Switching to Point-to-Point.")
                        estimation_method = o3d_reg.TransformationEstimationPointToPoint()
                        if source_instance_pcd_centered.has_normals(): source_instance_pcd_centered.normals.clear()
                        if target_pcd_model_centered.has_normals(): target_pcd_model_centered.normals.clear()  # Clear normals if switching to P2Point
                except Exception as e_norm:
                    print(f"    Error during normal estimation: {e_norm}. Switching to Point-to-Point ICP.")
                    estimation_method = o3d_reg.TransformationEstimationPointToPoint()
                    if source_instance_pcd_centered.has_normals(): source_instance_pcd_centered.normals.clear()
                    if target_pcd_model_centered.has_normals(): target_pcd_model_centered.normals.clear()

            if estimation_method is None:  # Should not happen if logic above is correct
                print(
                    f"Warning: Could not determine ICP estimation method for instance {inst_id}. Defaulting to Point-to-Point.")
                estimation_method = o3d_reg.TransformationEstimationPointToPoint()

            criteria_icp = o3d_reg.ICPConvergenceCriteria(
                relative_fitness=relative_fitness, relative_rmse=relative_rmse, max_iteration=max_iteration
            )
            initial_transform = np.identity(4)  # Initial guess: centered instance to centered model
            initial_transform[:3, :3] = R.from_euler('xyz', angles=[-90,0,0], degrees=True).as_matrix()
            # initial_transform[2,3] =  102.22797718e+00
            # print("initial_transform = ",initial_transform)
            # initial_transform = np.array([
            #     [9.94695735e-01, -2.20194751e-03, 1.02837471e-01, 0e+00],
            #     [8.52220541e-02, 5.77478552e-01, -8.11945641e-01, 102.22797718e+00],
            #     [-5.75985723e-02, 8.16402887e-01, 5.74603108e-01, 0e+00],
            #     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            # ])
            '''
            [[ 9.94695735e-01 -2.20194751e-03  1.02837471e-01 -3.52401870e+00]
             [ 8.52220541e-02  5.77478552e-01 -8.11945641e-01  2.22797718e+00]
             [-5.75985723e-02  8.16402887e-01  5.74603108e-01 -9.09074060e+00]
             [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
            '''
            print("  Using Identity matrix as initial transform (between centered clouds) for ICP.")

            print("  Executing ICP (Centered Instance vs Centered Model PCD)...")
            try:
                start_time = time.time()
                # ICP between centered instance (source) and centered sampled model (target)
                reg_result = o3d_reg.registration_icp(
                    source_instance_pcd_centered,  # Source: observed instance, centered
                    target_pcd_model_centered,  # Target: sampled model points, centered
                    threshold,
                    initial_transform,  # T_source_centered_to_target_centered
                    estimation_method,
                    criteria_icp
                )
                end_time = time.time()
                print(f"  ICP finished in {end_time - start_time:.3f} seconds.")
                print(f"  ICP Result: Fitness={reg_result.fitness:.4f}, RMSE={reg_result.inlier_rmse:.4f}")
                print("  Transformation from ICP (T_centered_instance_to_centered_model):")
                print(reg_result.transformation)

                # T_source_centered_to_target_centered (from ICP)
                T_s_centered_to_t_centered = reg_result.transformation

                T_translate_to_Cs_orig = np.eye(4)
                T_translate_to_Cs_orig[:3, 3] = source_instance_centroid_original

                T_translate_from_Ct_orig_model = np.eye(4)
                T_translate_from_Ct_orig_model[:3, 3] = -target_centroid_original_model

                T_s_centered_to_t_centered_inv = np.linalg.inv(T_s_centered_to_t_centered)

                estimated_pose = T_translate_to_Cs_orig @ T_s_centered_to_t_centered_inv @ T_translate_from_Ct_orig_model
                # This estimated_pose transforms the original model (e.g., STL) to the observed instance's position in the scene.

                print("\nCalculated Final Pose Matrix (transforms Original_Model_CAD -> Observed_Instance_in_Scene):")
                print(estimated_pose)
                estimated_poses[inst_id] = estimated_pose

                # Visualize ICP alignment (transformed model points vs. instance points)
                if args.visualize_pose:
                    # To visualize this, we transform the *original sampled model pcd* by the *final pose*
                    # This should align it with the *original preprocessed instance pcd*
                    transformed_model_pcd_for_instance_align = copy.deepcopy(
                        target_pcd_original_model)  # Use original (non-centered) sampled model
                    transformed_model_pcd_for_instance_align.transform(estimated_pose)

                    visualize_icp_alignment(
                        transformed_model_pcd_for_instance_align,  # Model points transformed to scene
                        instance_pcd_for_icp,  # Observed instance points in scene (preprocessed but not centered)
                        window_name=f"ICP Align (Model Pts in Scene vs Instance Pts) - Inst {inst_id}"
                    )

                # NEW VISUALIZATION: Transformed STL/CAD model in the original full scene
                if args.visualize_pose_in_scene:
                    # Use the original mesh if available, otherwise the original sampled point cloud
                    model_geometry_for_scene_viz = target_mesh if target_mesh.has_vertices() else target_pcd_original_model
                    if (isinstance(model_geometry_for_scene_viz,
                                   o3d.geometry.TriangleMesh) and not model_geometry_for_scene_viz.has_vertices()) or \
                            (isinstance(model_geometry_for_scene_viz,
                                        o3d.geometry.PointCloud) and not model_geometry_for_scene_viz.has_points()):
                        print("  Skipping scene visualization: original model geometry is empty.")
                    else:
                        visualize_transformed_model_in_scene(
                            original_scene_o3d_pcd,  # Full original scene PCD from HDF5
                            model_geometry_for_scene_viz,  # Original CAD model (mesh or pcd)
                            estimated_pose,  # Estimated pose (Original_Model_CAD -> Observed_Instance_in_Scene)
                            window_name=f"Final Model in Full Scene - Instance {inst_id}"
                        )

            except RuntimeError as e_icp:
                print(f"Error during ICP execution for instance {inst_id}: {e_icp}")
            except np.linalg.LinAlgError as e_linalg:
                print(f"Linear algebra error (e.g. singular matrix in inversion) for instance {inst_id}: {e_linalg}")
            except Exception as e_gen:
                print(f"General error during ICP processing for instance {inst_id}: {e_gen}")

    else:  # if num_instances_found == 0
        print("\nNo instances found or processed, skipping pose estimation visualization.")

    if args.save_results and estimated_poses:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        pose_filename = os.path.join(args.output_dir, f"estimated_poses_sample_{args.sample_index}_{timestamp}.npz")
        try:
            # Save poses: keys like 'instance_0', 'instance_1', etc.
            np.savez(pose_filename, **{f'instance_{k}': v for k, v in estimated_poses.items()})
            print(f"\nSaved estimated poses for {len(estimated_poses)} instances to {pose_filename}")
        except Exception as e_save:
            print(f"\nError saving poses: {e_save}")

    print("\nPose estimation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform Instance Segmentation and ICP Pose Estimation on HDF5 sample.')

    # --- Input/Output ---
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input_h5', type=str, default='./data/my_custom_dataset_h5_rgb/test_0.h5',
                          help='Input HDF5 file.')
    io_group.add_argument('--sample_index', type=int, default=0, help='Sample index in HDF5.')
    io_group.add_argument('--checkpoint', type=str, default="checkpoints_seg_pyg_ptconv_rgb/best_model.pth",
                          help='Semantic segmentation model checkpoint.')
    io_group.add_argument('--model_file', type=str, default="stp/cube.STL",
                          help='Target 3D model file (e.g., STL, PLY, OBJ) for ICP.')
    io_group.add_argument('--output_dir', type=str, default='./pose_estimation_results_cmdline',
                          help='Directory to save results.')

    # --- Data Processing ---
    data_proc_group = parser.add_argument_group('Data Processing')
    data_proc_group.add_argument('--num_points', type=int, default=2048,
                                 help='Expected number of points (informational, not strictly enforced by this script).')
    data_proc_group.add_argument('--no_rgb', action='store_true',
                                 help="Do not use RGB data even if available in HDF5 and model was trained with it.")
    data_proc_group.add_argument('--model_sample_points', type=int, default=2048*10,
                                 help='Number of points to sample from the CAD model if it is a mesh (for ICP target).')

    # --- Semantic Target Label & DBSCAN Parameters ---
    dbscan_group = parser.add_argument_group('Semantic Target & DBSCAN')
    dbscan_group.add_argument('--target_label_id', type=int, default=1, help='Target semantic label ID for DBSCAN.')
    dbscan_group.add_argument('--dbscan_eps', type=float, default=100,
                              help='DBSCAN epsilon (distance for neighborhood). Units should match point cloud.')
    dbscan_group.add_argument('--dbscan_min_samples', type=int, default=1000,
                              help='DBSCAN min_samples (min points in a neighborhood to form a core point).')

    # --- Instance Preprocessing Parameters ---
    pp_group = parser.add_argument_group('Instance Preprocessing (for ICP source cloud)')
    pp_group.add_argument('--preprocess_voxel_size', type=float, default=0.0,
                          help='Voxel size for instance downsampling. If > 0, enables. Units match point cloud.')
    pp_group.add_argument('--preprocess_sor_k', type=int, default=0,
                          help='Neighbors (k) for Statistical Outlier Removal (SOR). If > 0 (and std_ratio >0), enables SOR.')
    pp_group.add_argument('--preprocess_sor_std_ratio', type=float, default=0.0,
                          help='Std deviation ratio for SOR. If > 0 (and k > 0), enables SOR.')
    pp_group.add_argument('--preprocess_fps_n_points', type=int, default=0,
                          help='Target points for Farthest Point Sampling. If > 0, enables FPS. Applied after voxel/SOR.')

    # --- ICP Parameters ---
    icp_group = parser.add_argument_group('ICP Parameters')
    icp_group.add_argument('--icp_threshold', type=float, default=50,
                           help='ICP max_correspondence_distance threshold. Units match point cloud.')
    icp_group.add_argument('--icp_estimation_method', type=str, default='point_to_point',
                           choices=['point_to_point', 'point_to_plane'], help="ICP estimation method.")
    icp_group.add_argument('--icp_relative_fitness', type=float, default=1e-6,
                           help='ICP convergence: relative fitness change threshold.')
    icp_group.add_argument('--icp_relative_rmse', type=float, default=1e-6,
                           help='ICP convergence: relative RMSE change threshold.')
    icp_group.add_argument('--icp_max_iter', type=int, default=150,
                           help='ICP convergence: max iterations.')  # Was 100000, increased as per example. Default in O3D is often much lower (e.g. 30 or 50)
    icp_group.add_argument('--icp_min_points', type=int, default=100,
                           help='Min instance points required for ICP (after ALL preprocessing).')

    # --- Control Parameters ---
    ctrl_group = parser.add_argument_group('Control & Visualization')
    ctrl_group.add_argument('--no_cuda', action='store_true', help='Disable CUDA, use CPU.')
    ctrl_group.add_argument('--save_results', action='store_true', help='Save estimated poses to a .npz file.')
    ctrl_group.add_argument('--visualize_intermediate_pcds', action='store_true', default=False,
                            help='Visualize instance point clouds after preprocessing but before ICP.')
    ctrl_group.add_argument('--visualize_pose', action='store_true', default=True,
                            help='Visualize ICP alignment (transformed model points vs. observed instance points).')
    ctrl_group.add_argument('--visualize_pose_in_scene', action='store_true', default=True,  # NEW ARGUMENT
                            help='Visualize final transformed model (STL/CAD) within the original full scene point cloud.')

    # --- Model Architecture Parameters (Must match the trained model) ---
    model_arch_group = parser.add_argument_group('Model Architecture (match training config)')
    model_arch_group.add_argument('--num_classes', type=int, default=2,
                                  help='Number of semantic classes model was trained for.')
    model_arch_group.add_argument('--embed_dim', type=int, default=64,
                                  help='Initial embedding dimension for Point Transformer.')
    model_arch_group.add_argument('--k_neighbors', type=int, default=16, help='k for k-NN graph in Point Transformer.')
    model_arch_group.add_argument('--pt_hidden_dim', type=int, default=128, help='Point Transformer hidden dimension.')
    model_arch_group.add_argument('--pt_heads', type=int, default=4,
                                  help='Number of attention heads in Point Transformer.')
    model_arch_group.add_argument('--num_transformer_layers', type=int, default=2,
                                  help='Number of transformer layers in Point Transformer.')
    model_arch_group.add_argument('--dropout', type=float, default=0.3, help='Dropout rate used during model training.')

    args = parser.parse_args()
    main(args)

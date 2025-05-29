# _estimate_pose_from_h5_step6.py
# 版本: 统一语义分割模型加载逻辑，使其与 _evaluate_step5.py 类似。
# 对输入给语义模型的XYZ坐标进行归一化。
# 修改: 默认激活所有可视化过程。

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
    from model import PyG_PointTransformerSegModel  # 语义分割模型类

    try:
        from dataset import normalize_point_cloud_xyz as normalize_point_cloud_xyz_from_dataset

        print("Successfully imported normalize_point_cloud_xyz from dataset.py")
    except ImportError:
        print("normalize_point_cloud_xyz not found in dataset.py, will use local definition.")
        normalize_point_cloud_xyz_from_dataset = None
except ImportError as e:
    print(f"FATAL Error importing local modules (model.py or dataset.py): {e}")
    sys.exit(1)

# --- 导入 Open3D and Scikit-learn ---
try:
    import open3d as o3d
    from open3d.pipelines import registration as o3d_reg

    OPEN3D_AVAILABLE = True
except ImportError:
    print("FATAL Error: Open3D not found. Please install Open3D (pip install open3d).")
    sys.exit(1)
try:
    from sklearn.cluster import DBSCAN

    SKLEARN_AVAILABLE = True
except ImportError:
    print("FATAL Error: scikit-learn not found. Please install scikit-learn (pip install scikit-learn).")
    sys.exit(1)


# --- XYZ 坐标归一化函数 ---
def normalize_point_cloud_xyz_local(points_xyz):
    if points_xyz.shape[0] == 0: return points_xyz
    centroid = np.mean(points_xyz, axis=0)
    points_centered = points_xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
    if max_dist < 1e-6: return points_centered
    points_normalized = points_centered / max_dist
    return points_normalized.astype(np.float32)


if normalize_point_cloud_xyz_from_dataset is not None:
    normalize_xyz_for_semantic_model = normalize_point_cloud_xyz_from_dataset
    print("Using XYZ normalization function from dataset.py for semantic model input.")
else:
    normalize_xyz_for_semantic_model = normalize_point_cloud_xyz_local
    print("Using local XYZ normalization function for semantic model input (ensure it matches dataset.py).")


# --- 可视化函数 ---
def visualize_single_pcd(pcd, window_name="Point Cloud", point_color=[0.5, 0.5, 0.5]):
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    if not pcd.has_points(): print(f"Skipping visualization for {window_name}: No points."); return
    pcd_vis = copy.deepcopy(pcd)
    if not pcd_vis.has_colors(): pcd_vis.paint_uniform_color(point_color)
    print(f"\nDisplaying Point Cloud: {window_name} (Points: {len(pcd_vis.points)})")
    print("(Close the Open3D window to continue script execution...)")
    try:
        o3d.visualization.draw_geometries([pcd_vis], window_name=window_name)
        print(f"Visualization window '{window_name}' closed.")
    except Exception as e:
        print(f"Error during visualization of '{window_name}': {e}")


def visualize_icp_alignment(source_pcd_transformed_model, instance_pcd_observed, window_name="ICP Alignment Result"):
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    if not source_pcd_transformed_model.has_points() or not instance_pcd_observed.has_points():
        print(f"Skipping ICP alignment visualization for {window_name}: One or both point clouds are empty.")
        return
    instance_pcd_vis = copy.deepcopy(instance_pcd_observed)
    transformed_model_vis = copy.deepcopy(source_pcd_transformed_model)
    transformed_model_vis.paint_uniform_color([1, 0.706, 0])  # Yellow
    instance_pcd_vis.paint_uniform_color([0, 0.651, 0.929])  # Blue
    print(f"\nDisplaying ICP Alignment: {window_name}...")
    print("Yellow: Transformed Model Sampled Points | Blue: Observed Instance Points (preprocessed)")
    print("(Close the Open3D window to continue script execution...)")
    try:
        o3d.visualization.draw_geometries([transformed_model_vis, instance_pcd_vis], window_name=window_name)
        print(f"Alignment visualization window '{window_name}' closed.")
    except Exception as e:
        print(f"Error during visualization: {e}")


def visualize_transformed_model_in_scene(original_scene_pcd, target_model_geometry, estimated_pose,
                                         window_name="Transformed Model in Original Scene"):
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    scene_vis = copy.deepcopy(original_scene_pcd)
    model_vis = copy.deepcopy(target_model_geometry)
    model_vis.transform(estimated_pose)
    if not scene_vis.has_colors(): scene_vis.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
    if isinstance(model_vis, o3d.geometry.TriangleMesh):
        model_vis.paint_uniform_color([0, 1, 0])  # Green
        if not model_vis.has_vertex_normals(): model_vis.compute_vertex_normals()
    elif isinstance(model_vis, o3d.geometry.PointCloud):
        model_vis.paint_uniform_color([0, 1, 0])  # Green

    print(f"\nDisplaying: {window_name}")
    print("Scene Color (e.g., Gray/Original RGB): Original Full Scene | Green: Transformed Target Model")
    print("(Close the Open3D window to continue script execution...)")

    geometries_to_draw = []
    if scene_vis.has_points():  # Only add scene if it has points
        geometries_to_draw.append(scene_vis)
    else:
        print(f"Warning: Original scene for '{window_name}' has no points.")

    if (isinstance(model_vis, o3d.geometry.TriangleMesh) and model_vis.has_vertices()) or \
            (isinstance(model_vis, o3d.geometry.PointCloud) and model_vis.has_points()):
        geometries_to_draw.append(model_vis)
    else:
        print(f"Warning: Transformed model for '{window_name}' has no geometry, not drawing it.")

    if len(geometries_to_draw) > 0:  # Check if there's anything to draw
        try:
            o3d.visualization.draw_geometries(geometries_to_draw, window_name=window_name)
            print(f"Visualization window '{window_name}' closed.")
        except Exception as e:
            print(f"Error during visualization of '{window_name}': {e}")
    else:
        print(f"Not enough valid geometries to display for '{window_name}'.")


def visualize_semantic_segmentation(points_xyz, semantic_labels_np, num_classes, target_label_id,
                                    window_name="Semantic Segmentation Output"):
    """ 可视化语义分割结果 """
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    if points_xyz.shape[0] == 0: print(f"Skipping visualization for {window_name}: No points."); return

    print(f"\nDisplaying Semantic Segmentation: {window_name}")
    print("(Close the Open3D window to continue script execution...)")

    semantic_pcd = o3d.geometry.PointCloud()
    semantic_pcd.points = o3d.utility.Vector3dVector(points_xyz)

    # 为不同语义标签生成颜色
    # 你可以自定义一个颜色映射表
    max_label = np.max(semantic_labels_np) if semantic_labels_np.size > 0 else -1

    # 使用预定义的 distinct_colors 列表 (如果 _evaluate_step5.py 中有，或者在此处定义)
    distinct_colors = [
        [230 / 255, 25 / 255, 75 / 255], [60 / 255, 180 / 255, 75 / 255], [255 / 255, 225 / 255, 25 / 255],
        [0 / 255, 130 / 255, 200 / 255], [245 / 255, 130 / 255, 48 / 255], [145 / 255, 30 / 255, 180 / 255],
        [70 / 255, 240 / 255, 240 / 255], [240 / 255, 50 / 255, 230 / 255], [210 / 255, 245 / 255, 60 / 255],
        [250 / 255, 190 / 255, 212 / 255], [0 / 255, 128 / 255, 128 / 255], [220 / 255, 190 / 255, 255 / 255],
        [170 / 255, 110 / 255, 40 / 255], [255 / 255, 250 / 255, 200 / 255], [128 / 255, 0 / 255, 0 / 255],
        [128 / 255, 128 / 255, 0 / 255], [0 / 255, 0 / 255, 128 / 255], [128 / 255, 128 / 255, 128 / 255]
    ]

    num_distinct_colors = len(distinct_colors)
    colors_for_labels = np.zeros((max_label + 1, 3))

    for i in range(max_label + 1):
        if i == target_label_id:
            colors_for_labels[i] = [1, 0, 0]  # 目标类别用鲜红色
        else:
            colors_for_labels[i] = distinct_colors[i % num_distinct_colors]  # 其他类别循环使用颜色
            if i >= num_distinct_colors:  # 如果预定义颜色不够，用随机颜色补充（但尽量避免）
                colors_for_labels[i] = np.random.rand(3)

    if semantic_labels_np.size > 0:
        # Handle cases where semantic_labels_np might contain labels > max_label (should not happen with argmax)
        # or if max_label is -1 (empty semantic_labels_np, though we check points_xyz.shape[0] earlier)
        valid_labels_mask = semantic_labels_np <= max_label
        point_colors = np.zeros((semantic_labels_np.shape[0], 3))
        if np.any(valid_labels_mask):  # only apply colors if there are valid labels
            point_colors[valid_labels_mask] = colors_for_labels[semantic_labels_np[valid_labels_mask]]
        semantic_pcd.colors = o3d.utility.Vector3dVector(point_colors)
    else:
        semantic_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Default if no labels

    try:
        o3d.visualization.draw_geometries([semantic_pcd], window_name=window_name)
        print(f"Visualization window '{window_name}' closed.")
    except Exception as e:
        print(f"Error during visualization of '{window_name}': {e}")


# --- 主函数 ---
def main(args):
    print(f"Starting Pose Estimation from HDF5 at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    if not OPEN3D_AVAILABLE:  # Should have exited earlier, but double check
        print("Open3D is essential for this script. Exiting.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- 加载 HDF5 数据 ---
    print(f"Loading data from HDF5: {args.input_h5}, Sample: {args.sample_index}")
    points_xyz_original_from_h5 = None
    points_rgb_original_from_h5 = None
    if not os.path.exists(args.input_h5):
        print(f"FATAL Error: HDF5 file not found: {args.input_h5}");
        sys.exit(1)
    try:
        with h5py.File(args.input_h5, 'r') as f:
            if 'data' not in f: raise KeyError("'data' key missing in HDF5 file.")
            dset_data = f['data']
            dset_rgb = f.get('rgb', None)
            if not (0 <= args.sample_index < dset_data.shape[0]):
                raise IndexError(f"Sample index {args.sample_index} out of bounds (0 to {dset_data.shape[0] - 1}).")
            points_xyz_original_from_h5 = dset_data[args.sample_index].astype(np.float32)
            if dset_rgb is not None:
                if dset_rgb.shape[0] > args.sample_index and \
                        dset_rgb[args.sample_index].shape[0] == points_xyz_original_from_h5.shape[0]:
                    points_rgb_original_from_h5 = dset_rgb[args.sample_index].astype(np.uint8)
                else:
                    print(
                        f"Warning: RGB data in HDF5 has issues for sample index {args.sample_index}. Not using RGB for this sample.")
            else:
                print("No 'rgb' key found in HDF5. Not using RGB for this sample.")
    except Exception as e:
        print(f"FATAL Error loading HDF5: {e}");
        import traceback;
        traceback.print_exc();
        sys.exit(1)

    if points_xyz_original_from_h5 is None or points_xyz_original_from_h5.shape[0] == 0:
        print("FATAL Error: XYZ points data could not be loaded or is empty.");
        sys.exit(1)

    # 可视化原始加载的场景点云 (如果参数启用)
    if args.visualize_original_scene:
        original_scene_o3d_pcd_to_show = o3d.geometry.PointCloud()
        original_scene_o3d_pcd_to_show.points = o3d.utility.Vector3dVector(points_xyz_original_from_h5)
        if points_rgb_original_from_h5 is not None:
            original_scene_o3d_pcd_to_show.colors = o3d.utility.Vector3dVector(
                points_rgb_original_from_h5.astype(np.float64) / 255.0)
        visualize_single_pcd(original_scene_o3d_pcd_to_show, window_name=f"Original Scene - Sample {args.sample_index}")

    # --- 数据预处理: 为语义分割模型准备输入特征 ---
    points_xyz_normalized_for_semantic = normalize_xyz_for_semantic_model(points_xyz_original_from_h5)
    features_list_for_semantic = [points_xyz_normalized_for_semantic]
    if args.model_input_channels == 6:
        if points_rgb_original_from_h5 is not None:
            points_rgb_normalized_for_semantic = points_rgb_original_from_h5.astype(np.float32) / 255.0
            features_list_for_semantic.append(points_rgb_normalized_for_semantic)
        else:
            dummy_rgb = np.full((points_xyz_normalized_for_semantic.shape[0], 3), 0.5, dtype=np.float32)
            features_list_for_semantic.append(dummy_rgb)
    features_for_semantic_np = np.concatenate(features_list_for_semantic, axis=1)

    # --- 加载语义模型 ---
    print(f"\nLoading SEMANTIC model checkpoint from: {args.checkpoint_semantic}")
    if not os.path.isfile(args.checkpoint_semantic):
        print(f"FATAL Error: Semantic checkpoint not found: {args.checkpoint_semantic}");
        sys.exit(1)
    try:
        model_semantic = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
        checkpoint_data_semantic = torch.load(args.checkpoint_semantic, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint_data_semantic:
            model_semantic.load_state_dict(checkpoint_data_semantic['model_state_dict'])
        elif 'state_dict' in checkpoint_data_semantic:
            model_semantic.load_state_dict(checkpoint_data_semantic['state_dict'])
        else:
            model_semantic.load_state_dict(checkpoint_data_semantic)
        print("Semantic model weights loaded successfully.")
    except Exception as e:
        print(f"FATAL Error initializing or loading semantic model: {e}");
        import traceback;
        traceback.print_exc();
        sys.exit(1)
    model_semantic.eval()

    # --- 语义分割推理 ---
    print("\nPerforming semantic segmentation inference...")
    features_for_semantic_tensor = torch.from_numpy(features_for_semantic_np).float().unsqueeze(0).to(device)
    if features_for_semantic_tensor.shape[2] != args.model_input_channels:
        print(
            f"FATAL Error: Prepared semantic features dimension ({features_for_semantic_tensor.shape[2]}D) != model_input_channels ({args.model_input_channels}D).");
        sys.exit(1)
    with torch.no_grad():
        logits_semantic = model_semantic(features_for_semantic_tensor)
    pred_semantic_labels_np = torch.argmax(logits_semantic, dim=2).squeeze(0).cpu().numpy()
    unique_semantic, counts_semantic = np.unique(pred_semantic_labels_np, return_counts=True)
    print(f"  Predicted semantic label distribution: {dict(zip(unique_semantic, counts_semantic))}")

    # 可视化语义分割结果 (如果参数启用)
    if args.visualize_semantic_segmentation:
        # 使用原始（未归一化）的XYZ坐标进行可视化，以便观察真实尺度下的分割效果
        visualize_semantic_segmentation(points_xyz_original_from_h5, pred_semantic_labels_np, args.num_classes,
                                        args.target_label_id)

    # --- DBSCAN 聚类 ---
    print("\nPerforming DBSCAN clustering on original XYZ coordinates...")
    target_label_id = args.target_label_id
    target_mask_on_predictions = (pred_semantic_labels_np == target_label_id)
    points_xyz_for_clustering = points_xyz_original_from_h5[target_mask_on_predictions]

    num_instances_found = 0
    instance_points_dict = {}
    if points_xyz_for_clustering.shape[0] >= args.dbscan_min_samples:
        try:
            db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
            instance_labels_for_target_points = db.fit_predict(points_xyz_for_clustering)
            unique_instances = np.unique(
                instance_labels_for_target_points[instance_labels_for_target_points != -1])  # Exclude noise -1
            if args.visualize_dbscan_all_target_points and points_xyz_for_clustering.shape[0] > 0:
                # Visualize all points fed to DBSCAN, colored by instance label
                dbscan_vis_pcd = o3d.geometry.PointCloud()
                dbscan_vis_pcd.points = o3d.utility.Vector3dVector(points_xyz_for_clustering)
                max_instance_label = np.max(
                    instance_labels_for_target_points) if instance_labels_for_target_points.size > 0 else -1

                # Use distinct_colors, with -1 (noise) as gray
                colors_dbscan = np.random.rand(max_instance_label + 2, 3)  # +1 for 0-indexed, +1 for -1 index
                colors_dbscan[0] = [0.5, 0.5, 0.5]  # Color for noise (label -1, mapped to index 0)
                for lbl_idx in range(max_instance_label + 1):  # Colors for actual instances 0, 1, 2...
                    colors_dbscan[lbl_idx + 1] = distinct_colors[lbl_idx % len(distinct_colors)]

                dbscan_vis_pcd.colors = o3d.utility.Vector3dVector(
                    colors_dbscan[instance_labels_for_target_points + 1])  # Shift labels to be non-negative indices
                visualize_single_pcd(dbscan_vis_pcd, window_name=f"DBSCAN Output (Label {target_label_id} points)")

            unique_instances.sort()
            num_instances_found = len(unique_instances)
            print(f"Clustering found {num_instances_found} potential instances with label {target_label_id}.")
            for inst_id in unique_instances:
                instance_mask_local = (instance_labels_for_target_points == inst_id)
                instance_points_dict[inst_id] = points_xyz_for_clustering[instance_mask_local]
                print(f"  Instance {inst_id}: {instance_points_dict[inst_id].shape[0]} points (original scale)")
        except Exception as e:
            print(f"Error during DBSCAN clustering: {e}");
            num_instances_found = 0;
            instance_points_dict = {}
    else:
        print(
            f"Not enough points for DBSCAN (semantic label {target_label_id}): required {args.dbscan_min_samples}, found {points_xyz_for_clustering.shape[0]}.")

    # --- 加载目标 CAD/扫描模型 ---
    print(f"\nLoading target CAD/model file from: {args.model_file}")
    target_mesh = o3d.geometry.TriangleMesh()
    target_pcd_original_model_scale = o3d.geometry.PointCloud()
    target_centroid_original_model_scale = np.zeros(3)
    if not os.path.exists(args.model_file):
        print(f"FATAL Error: Target model file not found: {args.model_file}");
        sys.exit(1)
    try:
        temp_mesh = o3d.io.read_triangle_mesh(args.model_file)
        if temp_mesh.has_vertices():
            target_mesh = temp_mesh
            target_pcd_original_model_scale = target_mesh.sample_points_uniformly(
                number_of_points=args.model_sample_points)
            if not target_pcd_original_model_scale.has_points(): raise ValueError("Sampling from mesh failed.")
        else:
            target_pcd_original_model_scale = o3d.io.read_point_cloud(args.model_file)
            if not target_pcd_original_model_scale.has_points(): raise ValueError(
                "Target model file (as PCD) has no points.")
        target_centroid_original_model_scale = target_pcd_original_model_scale.get_center()
        target_pcd_model_centered_for_icp = copy.deepcopy(target_pcd_original_model_scale)
        target_pcd_model_centered_for_icp.translate(-target_centroid_original_model_scale)
        if args.visualize_cad_model:  # Visualize the loaded (and sampled if mesh) CAD model
            visualize_single_pcd(target_pcd_original_model_scale, window_name="Loaded CAD Model (Sampled for ICP)")
    except Exception as e:
        print(f"FATAL Error loading or processing target CAD/model file: {e}");
        sys.exit(1)

    # --- Configure ICP Parameters ---
    print("\nConfiguring ICP parameters...")
    threshold_icp = args.icp_threshold
    method_str_icp = args.icp_estimation_method

    estimated_poses = {}
    if num_instances_found > 0:
        print("\nPerforming ICP for each found instance...")
        for inst_id, instance_points_original_scale_np in instance_points_dict.items():
            print(f"\nProcessing Instance ID: {inst_id} (Initial points: {instance_points_original_scale_np.shape[0]})")
            instance_pcd_original_scale = o3d.geometry.PointCloud()
            instance_pcd_original_scale.points = o3d.utility.Vector3dVector(instance_points_original_scale_np)
            instance_pcd_for_icp = copy.deepcopy(instance_pcd_original_scale)

            # Instance Preprocessing
            if args.preprocess_voxel_size > 0 and instance_pcd_for_icp.has_points():
                print(f"    Applying Voxel Downsampling (voxel_size={args.preprocess_voxel_size})")
                instance_pcd_for_icp = instance_pcd_for_icp.voxel_down_sample(args.preprocess_voxel_size)
                if not instance_pcd_for_icp.has_points(): print(
                    f"  Instance {inst_id} empty after Voxel. Skipping."); continue
            if args.preprocess_sor_k > 0 and args.preprocess_sor_std_ratio > 0 and instance_pcd_for_icp.has_points():
                print(f"    Applying SOR (k={args.preprocess_sor_k}, std_ratio={args.preprocess_sor_std_ratio})")
                sor_pcd, _ = instance_pcd_for_icp.remove_statistical_outlier(nb_neighbors=args.preprocess_sor_k,
                                                                             std_ratio=args.preprocess_sor_std_ratio)
                if sor_pcd.has_points():
                    instance_pcd_for_icp = sor_pcd
                else:
                    print("    Warning: SOR removed all points. Using points before SOR.")
                if not instance_pcd_for_icp.has_points(): print(
                    f"  Instance {inst_id} empty after SOR. Skipping."); continue
            if args.preprocess_fps_n_points > 0 and instance_pcd_for_icp.has_points():
                if len(instance_pcd_for_icp.points) > args.preprocess_fps_n_points:
                    print(f"    Applying FPS to {args.preprocess_fps_n_points} points")
                    instance_pcd_for_icp = instance_pcd_for_icp.farthest_point_down_sample(args.preprocess_fps_n_points)
                if not instance_pcd_for_icp.has_points(): print(
                    f"  Instance {inst_id} empty after FPS. Skipping."); continue

            if args.visualize_intermediate_pcds and instance_pcd_for_icp.has_points():
                visualize_single_pcd(instance_pcd_for_icp,
                                     window_name=f"Preprocessed Instance {inst_id} for ICP ({len(instance_pcd_for_icp.points)} pts)")

            if len(instance_pcd_for_icp.points) < args.icp_min_points:
                print(
                    f"  Skipping ICP for Inst {inst_id}: Not enough points after preprocessing ({len(instance_pcd_for_icp.points)} < {args.icp_min_points}).");
                continue

            instance_centroid_for_icp = instance_pcd_for_icp.get_center()
            instance_pcd_centered_for_icp = copy.deepcopy(instance_pcd_for_icp)
            instance_pcd_centered_for_icp.translate(-instance_centroid_for_icp)

            estimation_method_icp = o3d_reg.TransformationEstimationPointToPoint()
            if method_str_icp.lower() == 'point_to_plane':
                normal_radius_icp = threshold_icp * 2.0;
                normal_radius_icp = max(1e-3, normal_radius_icp)
                try:
                    instance_pcd_centered_for_icp.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_icp, max_nn=30))
                    if not target_pcd_model_centered_for_icp.has_normals():
                        target_pcd_model_centered_for_icp.estimate_normals(
                            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_icp, max_nn=30))
                    if instance_pcd_centered_for_icp.has_normals() and target_pcd_model_centered_for_icp.has_normals():
                        estimation_method_icp = o3d_reg.TransformationEstimationPointToPlane()
                except Exception:
                    pass

            criteria_icp = o3d_reg.ICPConvergenceCriteria(relative_fitness=args.icp_relative_fitness,
                                                          relative_rmse=args.icp_relative_rmse,
                                                          max_iteration=args.icp_max_iter)
            initial_transform_icp = np.identity(4)
            try:
                reg_result = o3d_reg.registration_icp(instance_pcd_centered_for_icp, target_pcd_model_centered_for_icp,
                                                      threshold_icp, initial_transform_icp, estimation_method_icp,
                                                      criteria_icp)
                T_centered_s_to_centered_t = reg_result.transformation
                T_world_to_instance_centroid = np.eye(4);
                T_world_to_instance_centroid[:3, 3] = instance_centroid_for_icp
                T_model_centroid_to_world = np.eye(4);
                T_model_centroid_to_world[:3, 3] = -target_centroid_original_model_scale
                final_estimated_pose = T_world_to_instance_centroid @ T_centered_s_to_centered_t @ T_model_centroid_to_world
                estimated_poses[inst_id] = final_estimated_pose
                print(f"  ICP for Inst {inst_id}: Fitness={reg_result.fitness:.4f}, RMSE={reg_result.inlier_rmse:.4f}")

                if args.visualize_pose:
                    transformed_cad_pcd_for_viz = copy.deepcopy(target_pcd_original_model_scale);
                    transformed_cad_pcd_for_viz.transform(final_estimated_pose)
                    visualize_icp_alignment(transformed_cad_pcd_for_viz, instance_pcd_for_icp,
                                            window_name=f"ICP Align - Inst {inst_id}")
                if args.visualize_pose_in_scene:
                    model_geometry_for_scene_viz = target_mesh if target_mesh.has_vertices() else target_pcd_original_model_scale
                    visualize_transformed_model_in_scene(original_scene_o3d_pcd, model_geometry_for_scene_viz,
                                                         final_estimated_pose,
                                                         window_name=f"Final CAD in Scene - Inst {inst_id}")
            except Exception as e_icp_run:
                print(f"Error during ICP run for instance {inst_id}: {e_icp_run}")
    else:
        print("\nNo instances for ICP.")

    if args.save_results and estimated_poses:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        pose_filename = os.path.join(args.output_dir, f"estimated_poses_h5sample_{args.sample_index}_{timestamp}.npz")
        try:
            np.savez(pose_filename, **{f'instance_{k}': v for k, v in estimated_poses.items()}); print(
                f"\nSaved poses to {pose_filename}")
        except Exception as e_save:
            print(f"\nError saving poses: {e_save}")
    print("\nPose estimation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform Instance Segmentation and ICP Pose Estimation on HDF5 sample.')
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input_h5', type=str, default='./data/testla_part1_h5/test_0.h5', help='Input HDF5 file.')
    io_group.add_argument('--sample_index', type=int, default=0, help='Sample index in HDF5.')
    io_group.add_argument('--checkpoint_semantic', type=str,
                          default="checkpoints_seg_tesla_part1_normalized/best_model.pth",
                          help='Semantic segmentation model checkpoint.')
    io_group.add_argument('--model_file', type=str, default="stp/part1_rude.STL",
                          help='Target 3D model file (e.g., STL, PLY, OBJ) for ICP.')
    io_group.add_argument('--output_dir', type=str, default='./pose_estimation_results_cmdline',
                          help='Directory to save results.')

    model_config_group = parser.add_argument_group('Semantic Model Configuration & Architecture (match training!)')
    model_config_group.add_argument('--num_classes', type=int, default=2,
                                    help='Number of semantic classes model was trained for (passed to Model).')
    model_config_group.add_argument('--model_input_channels', type=int, default=6, choices=[3, 6],
                                    help="Input channels expected by the semantic model (3 for XYZ, 6 for XYZRGB). This dictates feature preparation.")
    model_config_group.add_argument('--k_neighbors', type=int, default=16, help='(Model Arch) k for k-NN graph.')
    model_config_group.add_argument('--embed_dim', type=int, default=64,
                                    help='(Model Arch) Initial embedding dimension.')
    model_config_group.add_argument('--pt_hidden_dim', type=int, default=128,
                                    help='(Model Arch) Point Transformer hidden dimension.')
    model_config_group.add_argument('--pt_heads', type=int, default=4, help='(Model Arch) Number of attention heads.')
    model_config_group.add_argument('--num_transformer_layers', type=int, default=2,
                                    help='(Model Arch) Number of transformer layers.')
    model_config_group.add_argument('--dropout', type=float, default=0.3,
                                    help='(Model Arch) Dropout rate used during model training.')

    data_proc_group = parser.add_argument_group('Other Data Processing Parameters')
    data_proc_group.add_argument('--model_sample_points', type=int, default=2048,
                                 help='Number of points to sample from the CAD model for ICP target.')

    dbscan_group = parser.add_argument_group('Semantic Target & DBSCAN')
    dbscan_group.add_argument('--target_label_id', type=int, default=1, help='Target semantic label ID for DBSCAN.')
    dbscan_group.add_argument('--dbscan_eps', type=float, default=100,
                              help='DBSCAN epsilon (units match original point cloud scale).')
    dbscan_group.add_argument('--dbscan_min_samples', type=int, default=1000, help='DBSCAN min_samples.')

    pp_group = parser.add_argument_group('Instance Preprocessing (units match original scale)')
    pp_group.add_argument('--preprocess_voxel_size', type=float, default=0.01,
                          help='Voxel size for instance downsampling (0 to disable).')
    pp_group.add_argument('--preprocess_sor_k', type=int, default=20, help='Neighbors for SOR (0 to disable).')
    pp_group.add_argument('--preprocess_sor_std_ratio', type=float, default=1.0,
                          help='Std ratio for SOR (0 to disable).')
    pp_group.add_argument('--preprocess_fps_n_points', type=int, default=1024,
                          help='Target points for FPS (0 to disable).')

    icp_group = parser.add_argument_group('ICP Parameters (units match original scale)')
    icp_group.add_argument('--icp_threshold', type=float, default=0.02, help='ICP max_correspondence_distance.')
    icp_group.add_argument('--icp_estimation_method', type=str, default='point_to_plane',
                           choices=['point_to_point', 'point_to_plane'], help="ICP estimation method.")
    icp_group.add_argument('--icp_relative_fitness', type=float, default=1e-7,
                           help='ICP convergence: relative fitness.')
    icp_group.add_argument('--icp_relative_rmse', type=float, default=1e-7, help='ICP convergence: relative RMSE.')
    icp_group.add_argument('--icp_max_iter', type=int, default=200, help='ICP convergence: max iterations.')
    icp_group.add_argument('--icp_min_points', type=int, default=20,
                           help='Min instance points for ICP (after preprocessing).')

    ctrl_group = parser.add_argument_group('Control & Visualization')
    ctrl_group.add_argument('--no_cuda', action='store_true', help='Disable CUDA, use CPU.')
    ctrl_group.add_argument('--save_results', action='store_true', help='Save estimated poses to a .npz file.')
    # --- MODIFIED: Set visualization flags to True by default ---
    ctrl_group.add_argument('--visualize_original_scene', action='store_true', default=True,
                            help='Visualize the original scene loaded from HDF5.')
    ctrl_group.add_argument('--visualize_semantic_segmentation', action='store_true', default=True,
                            help='Visualize the output of semantic segmentation.')
    ctrl_group.add_argument('--visualize_dbscan_all_target_points', action='store_true', default=True,
                            help='Visualize all points of target semantic class colored by DBSCAN instance labels (includes noise).')
    ctrl_group.add_argument('--visualize_cad_model', action='store_true', default=True,
                            help='Visualize the loaded CAD model (sampled points if mesh).')
    ctrl_group.add_argument('--visualize_intermediate_pcds', action='store_true', default=True,
                            help='Visualize instance point clouds after preprocessing but before ICP.')
    ctrl_group.add_argument('--visualize_pose', action='store_true', default=True,
                            help='Visualize ICP alignment (transformed model points vs. observed instance points).')
    ctrl_group.add_argument('--visualize_pose_in_scene', action='store_true', default=True,
                            help='Visualize final transformed model (STL/CAD) within the original full scene point cloud.')

    args = parser.parse_args()
    main(args)
# _estimate_pose_from_h5_args_icp.py
# 加载 HDF5 样本，执行语义分割+聚类，然后对每个实例进行 ICP 姿态估计。
# 所有 ICP 参数均通过命令行参数进行配置。

import torch
import numpy as np
import argparse
import time
import os
import datetime
import sys
import h5py
import copy # 用于深拷贝 Open3D 几何体

# --- 导入本地模块 ---
try:
    from model import PyG_PointTransformerSegModel # 语义分割模型类
except ImportError as e: print(f"FATAL Error importing model: {e}"); sys.exit(1)

# --- 导入 Open3D 和 Scikit-learn ---
try:
    import open3d as o3d
    # 导入 ICP 相关模块
    from open3d.pipelines import registration as o3d_reg
    OPEN3D_AVAILABLE = True
except ImportError: print("FATAL Error: Open3D not found."); sys.exit(1)
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError: print("FATAL Error: scikit-learn not found."); sys.exit(1)


# --- 预处理函数 (与原版本相同) ---
def preprocess_h5_data(h5_file_path, sample_index, num_target_points, no_rgb, normalize):
    """
    加载 HDF5 数据，提取点云和 RGB (如果需要)，并可选地进行归一化。

    Args:
        h5_file_path (str): HDF5 文件路径。
        sample_index (int): 要加载的样本索引。
        num_target_points (int): 期望的点数（当前版本使用 HDF5 中的实际点数）。
        no_rgb (bool): 是否不加载/使用 RGB 数据。
        normalize (bool): 是否对点云 XYZ 坐标进行归一化。

    Returns:
        tuple: 包含 (features_tensor, points_processed_np)。
               features_tensor (torch.Tensor): 模型输入特征张量 (1, N, C)。
               points_processed_np (np.ndarray): 处理后的 XYZ 点云 NumPy 数组 (N, 3)。
               如果加载或处理失败，返回 (None, None)。
    """
    print(f"Loading data from HDF5: {h5_file_path}, Sample: {sample_index}")
    if not os.path.exists(h5_file_path): raise FileNotFoundError(f"HDF5 not found: {h5_file_path}")
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'data' not in f: raise KeyError("'data' key missing")
            dset_data = f['data']; dset_rgb = f['rgb'] if not no_rgb and 'rgb' in f else None
            if not (0 <= sample_index < dset_data.shape[0]): raise IndexError("Sample index out of bounds.")
            points_xyz_np = dset_data[sample_index].astype(np.float32)
            points_rgb_np = dset_rgb[sample_index].astype(np.uint8) if dset_rgb is not None else None

            points_processed_np = np.copy(points_xyz_np) # Keep original points for potential later use if normalize is for model input only

            if normalize:
                # Note: Normalization might be for model input features, not for ICP points depending on workflow.
                # Adjust this part if normalize should affect points_processed_np directly.
                centroid = np.mean(points_processed_np, axis=0);
                points_normalized = points_processed_np - centroid
                max_dist = np.max(np.sqrt(np.sum(points_normalized ** 2, axis=1)))
                if max_dist > 1e-6:
                     points_normalized /= max_dist
                print("Applied normalization (center + unit sphere) for model input.")
                features_list = [points_normalized] # Use normalized XYZ as model input
            else:
                 features_list = [points_processed_np] # Use original XYZ as model input


            model_input_channels = 3
            if not no_rgb:
                model_input_channels = 6
                if points_rgb_np is not None: features_list.append(points_rgb_np.astype(np.float32) / 255.0)
                else: features_list.append(np.full((points_processed_np.shape[0], 3), 0.5, dtype=np.float32))

            features_np = np.concatenate(features_list, axis=1)

            if features_np.shape[1] != model_input_channels: raise ValueError(f"Feature dim mismatch: Got {features_np.shape[1]}D, Expected {model_input_channels}D after processing.")
            if features_np.shape[0] != points_processed_np.shape[0]: raise ValueError("Feature and point counts mismatch.")

            features_tensor = torch.from_numpy(features_np).float().unsqueeze(0)
            # Return original points_processed_np if normalize is only for model input
            return features_tensor, points_xyz_np # Return original points for ICP later


    except Exception as e: print(f"Error loading/preprocessing HDF5: {e}"); return None, None


# --- 可视化函数: 显示 ICP 对齐结果 ---
def visualize_icp_alignment(source_pcd_transformed, target_pcd, window_name="ICP Alignment Result"):
    """可视化对齐后的源点云（实例）和目标点云（模型）。"""
    if not OPEN3D_AVAILABLE: return
    # 给目标和源点云不同颜色以区分
    target_pcd_vis = copy.deepcopy(target_pcd)
    source_pcd_transformed_vis = copy.deepcopy(source_pcd_transformed)
    target_pcd_vis.paint_uniform_color([0, 0.651, 0.929])  # 蓝色 (模型)
    source_pcd_transformed_vis.paint_uniform_color([1, 0.706, 0]) # 黄色 (实例)

    print(f"\nDisplaying ICP Alignment for {window_name}...")
    print("Blue: Target Model | Yellow: Aligned Instance")
    print("(Close the window to continue...)")
    o3d.visualization.draw_geometries([source_pcd_transformed_vis, target_pcd_vis], window_name=window_name)
    print("Alignment visualization window closed.")


# --- 主函数 ---
def main(args):
    print(f"Starting Pose Estimation from HDF5 at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")
    if not SKLEARN_AVAILABLE: print("FATAL: scikit-learn required."); sys.exit(1)
    if not OPEN3D_AVAILABLE: print("FATAL: Open3D required."); sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"); print(f"Using device: {device}")
    if args.save_results: os.makedirs(args.output_dir, exist_ok=True)

    # --- 加载语义模型 ---
    print(f"\nLoading SEMANTIC model checkpoint from: {args.checkpoint}")
    # ... (模型加载逻辑不变) ...
    if not os.path.exists(args.checkpoint): raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model_input_channels = 3 if args.no_rgb else 6; print(f"Initializing model for {model_input_channels}D input...")
    try: model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device); print("Model structure initialized.")
    except Exception as e: print(f"Error init model: {e}"); return
    try: checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except Exception as e: print(f"Error loading checkpoint file: {e}"); return
    try:
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    except Exception as e: print(f"Error loading state dict: {e}"); return
    model.eval()


    # --- 加载和预处理 HDF5 数据 ---
    # preprocess_h5_data 返回原始 XYZ 坐标 points_original_np
    features_tensor, points_original_np = preprocess_h5_data(
        args.input_h5, args.sample_index, args.num_points, args.no_rgb, args.normalize # normalize flag affects feature_tensor, points_original_np is always original XYZ
    )
    if features_tensor is None or points_original_np is None:
         print("Exiting due to preprocessing error."); return


    # --- 语义分割推理 ---
    print("\nPerforming semantic segmentation inference...")
    features_tensor = features_tensor.to(device)
    with torch.no_grad(): logits = model(features_tensor)
    pred_semantic_labels_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy() # (N,)
    print("Semantic prediction complete.")
    unique_semantic, counts_semantic = np.unique(pred_semantic_labels_np, return_counts=True)
    print(f"  Predicted semantic label distribution: {dict(zip(unique_semantic, counts_semantic))}")


    # --- DBSCAN 聚类 ---
    print("\nPerforming DBSCAN clustering...")
    screw_label_id = args.screw_label_id
    screw_mask = (pred_semantic_labels_np == screw_label_id)
    screw_points_xyz = points_original_np[screw_mask] # Use original points for clustering
    instance_labels_full = np.full_like(pred_semantic_labels_np, -1, dtype=np.int64)
    num_instances_found = 0
    instance_points_dict = {} # 存储每个实例的点云 {inst_id: points_array}

    if screw_points_xyz.shape[0] >= args.dbscan_min_samples:
        db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
        instance_labels_screw_points = db.fit_predict(screw_points_xyz)
        instance_labels_full[screw_mask] = instance_labels_screw_points
        unique_instances = np.unique(instance_labels_screw_points[instance_labels_screw_points != -1])
        unique_instances.sort() # Sort instance IDs for consistent processing order
        num_instances_found = len(unique_instances)
        print(f"Clustering found {num_instances_found} potential instances.")
        # 将每个实例的点存储起来
        for inst_id in unique_instances:
            instance_mask_local = (instance_labels_screw_points == inst_id)
            instance_points_dict[inst_id] = screw_points_xyz[instance_mask_local]
            print(f"  Instance {inst_id}: {instance_points_dict[inst_id].shape[0]} points")
    else:
        print(f"Not enough points predicted as target (label {screw_label_id}) for DBSCAN.")


    # --- 加载目标 CAD/扫描模型 ---
    print(f"\nLoading target model file from: {args.model_file}")
    if not os.path.exists(args.model_file): raise FileNotFoundError(f"Target model file not found: {args.model_file}")
    try:
        # 尝试作为三角网格加载
        target_mesh = o3d.io.read_triangle_mesh(args.model_file)
        target_pcd_original = None # 用于保存原始模型点云 (如果从网格采样) 或直接加载点云
        target_centroid_original = np.zeros(3) # 用于保存原始模型中心

        if not target_mesh.has_vertices(): # 如果失败或为空，尝试作为点云加载
             print("  Could not load as mesh or mesh is empty, attempting to load as point cloud...")
             target_pcd = o3d.io.read_point_cloud(args.model_file)
             if not target_pcd.has_points(): raise ValueError("Target model file contains no points.")
             target_pcd_original = copy.deepcopy(target_pcd) # 保存原始点云
             target_centroid_original = target_pcd_original.get_center() # 保存原始中心
             # 可选：计算法线 (如果使用 P2Plane ICP)，这里先不计算
        else:
             # 如果是网格，需要采样成点云才能用于点对点 ICP
             print(f"  Loaded mesh with {len(target_mesh.vertices)} vertices. Sampling points for ICP...")
             # 采样前保存原始中心 (使用 BBox 中心作为近似)
             mesh_bbox = target_mesh.get_axis_aligned_bounding_box()
             target_centroid_original = mesh_bbox.get_center()

             # 采样点数可以调整，例如与实例点云数量相当，或者更多
             num_model_points_to_sample = max(args.num_points, 4096*2) # 至少采样 5000 点，或与输入点数一致
             target_pcd = target_mesh.sample_points_uniformly(number_of_points=num_model_points_to_sample)
             # 注意：采样后点云的中心可能与原始网格中心略有偏差，但通常用于 ICP 影响不大
             target_pcd_original = copy.deepcopy(target_pcd) # 保存采样后的点云


             # 可选：计算法线 (如果使用 P2Plane)
             # if args.icp_estimation_method.lower() == 'point_to_plane':
             #    print("  Estimating normals for target model for Point-to-Plane ICP...")
             #    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.icp_threshold * 2, max_nn=30))


        print(f"Target model loaded/sampled as point cloud with {len(target_pcd.points)} points.")
        print(f"  Original Target Model Center: [{target_centroid_original[0]:.3f},{target_centroid_original[1]:.3f},{target_centroid_original[2]:.3f}]")


        # --- 预处理目标模型: 中心化 ---
        # 将目标模型移到原点，ICP 对齐时更容易处理
        target_pcd_centered = copy.deepcopy(target_pcd)
        # 使用保存的原始中心进行平移
        target_pcd_centered.translate(-target_centroid_original)
        print(f"  Target model centered around origin for ICP.")

    except Exception as e:
        print(f"Error loading or processing target model file: {e}"); return


    # --- 配置 ICP 参数 (从命令行参数获取) ---
    print("\nConfiguring ICP parameters from command-line arguments:")

    # 1. max_distance (对应点对的最大距离阈值):
    #    只有距离小于此值的源点和目标点才会被视为有效对应点。单位应与你的点云坐标单位一致。
    #    需要根据点云的噪声水平、分辨率以及预期的对齐精度进行调整。
    threshold = args.icp_threshold
    print(f"  ICP max_distance (threshold): {threshold}")

    # 2. init (初始变换矩阵): ICP 的起始变换。
    #    一个 4x4 的 NumPy 数组，表示源点云相对于目标点云的初始旋转和平移。
    #    良好的初始值对 ICP 收敛到正确解至关重要，尤其当点云初始位置相差较大时。
    #    这里默认使用单位矩阵，这假设中心化后的源和目标已经粗略对齐。
    #    更复杂的初始变换（如从文件加载或通过其他方法计算）需要额外的参数和逻辑。
    trans_init = np.identity(4)
    print(f"  ICP initial_transform (init):\n{trans_init}")


    # 3. estimation_method (变换估计方法): 根据对应点对计算变换的方法。
    #    'point_to_point': 点对点方法，最小化对应点间距离平方和。不需要法线。
    #    'point_to_plane': 点对面方法，最小化源点到目标点切平面距离平方和。需要准确的法线。
    #    根据点云特性和需求选择。
    if args.icp_estimation_method.lower() == 'point_to_point':
        estimation_method = o3d_reg.TransformationEstimationPointToPoint()
        print(f"  ICP estimation_method: Point-to-Point")
    elif args.icp_estimation_method.lower() == 'point_to_plane':
        estimation_method = o3d_reg.TransformationEstimationPointToPlane()
        print(f"  ICP estimation_method: Point-to-Plane")
        # 警告用户点对面需要法线，并建议计算法线
        print("  Warning: Using Point-to-Plane ICP. Please ensure point clouds have accurate normals.")
        # Note: Normal calculation code for source/target is commented out above. Uncomment and configure if using P2Plane.
    else:
        raise ValueError(f"Unknown ICP estimation method: {args.icp_estimation_method}. Choose 'point_to_point' or 'point_to_plane'.")


    # 4. criteria (收敛标准): 控制 ICP 何时停止迭代。
    #    relative_fitness: Fitness 相对变化阈值：当连续两次迭代 Fitness 相对变化小于此值时停止。
    #    relative_rmse: RMSE 相对变化阈值：当连续两次迭代 RMSE 相对变化小于此值时停止。
    #    max_iteration: 最大迭代次数：达到此次数后强制停止。防止无限循环。
    relative_fitness = args.icp_relative_fitness
    relative_rmse = args.icp_relative_rmse
    max_iteration = args.icp_max_iter

    criteria = o3d_reg.ICPConvergenceCriteria(
         relative_fitness=relative_fitness,
         relative_rmse=relative_rmse,
         max_iteration=max_iteration
    )
    print(f"  ICP convergence_criteria:")
    print(f"    relative_fitness: {criteria.relative_fitness}")
    print(f"    relative_rmse: {criteria.relative_rmse}")
    print(f"    max_iteration: {criteria.max_iteration}")


    # --- 对每个找到的实例执行 ICP 姿态估计 ---
    estimated_poses = {} # 存储每个实例的姿态 {inst_id: 4x4_matrix}
    if num_instances_found > 0:
        print("\nPerforming ICP for each found instance...")

        for inst_id, instance_points_np in instance_points_dict.items():
            print(f"\nProcessing Instance ID: {inst_id}")
            if instance_points_np.shape[0] < args.icp_min_points: # 使用命令行参数 icp_min_points
                 print(f"  Skipping ICP: Instance has only {instance_points_np.shape[0]} points (min required: {args.icp_min_points}).")
                 continue

            # 创建源点云 (实例)
            source_pcd_original = o3d.geometry.PointCloud()
            source_pcd_original.points = o3d.utility.Vector3dVector(instance_points_np)

            # 获取源点云的原始中心
            source_centroid_original = source_pcd_original.get_center()
            print(f"  Original Instance Center: [{source_centroid_original[0]:.3f},{source_centroid_original[1]:.3f},{source_centroid_original[2]:.3f}]")


            # --- 预处理源点云: 中心化 ---
            # 注意：这里中心化是为了 ICP 对齐，最终的变换矩阵需要考虑这个中心化偏移
            source_pcd_centered = copy.deepcopy(source_pcd_original)
            source_pcd_centered.translate(-source_centroid_original) # 使用原始中心进行平移
            print(f"  Instance centered for ICP.")

            # 可选：计算法线 (如果使用 P2Plane)
            # if args.icp_estimation_method.lower() == 'point_to_plane':
            #    print("  Estimating normals for instance for Point-to-Plane ICP...")
            #    source_pcd_centered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.icp_threshold * 2, max_nn=30))


            # 执行 ICP
            print(f"  Running ICP with configured parameters...")
            reg_result = o3d_reg.registration_icp(
                source_pcd_centered, # 已中心化的源点云 (实例)
                target_pcd_centered, # 已中心化的目标点云 (模型)
                threshold,           # 对应点对最大距离阈值 (来自命令行参数)
                trans_init,          # 初始变换矩阵 (默认为单位矩阵)
                estimation_method,   # 变换估计方法 (来自命令行参数)
                criteria             # 收敛标准 (来自命令行参数)
            )

            # reg_result 包含 fitness, inlier_rmse, transformation
            print(f"  ICP Result: Fitness={reg_result.fitness:.4f}, RMSE={reg_result.inlier_rmse:.4f}")
            print("  Estimated Transformation Matrix (source_centered -> target_centered):")
            print(reg_result.transformation)

            # --- 计算最终姿态 (将原始模型变换到原始实例点云位置) ---
            # ICP 得到的 reg_result.transformation (记为 T) 是将中心化的 source_pcd 变换到中心化的 target_pcd
            # 即 T * (Source_original - C_source) ≈ (Target_original - C_target)
            # 我们要找的最终姿态 P 是将 Target_original 变换到 Source_original 的矩阵
            # 即 Source_original ≈ P * Target_original
            # 从 ICP 结果反推 P:
            # T^-1 * (Target_original - C_target) ≈ Source_original - C_source
            # T^-1 * Target_original - T^-1 * C_target ≈ Source_original - C_source
            # Source_original ≈ T^-1 * Target_original - T^-1 * C_target + C_source
            # 所以最终姿态矩阵 P 的旋转部分是 T 的逆的旋转部分，平移部分是 C_source - T_inv @ C_target
            # 其中 T 是 reg_result.transformation，T_inv 是 T 的逆，C_source 是 source_centroid_original，C_target 是 target_centroid_original

            T_icp = reg_result.transformation
            T_icp_inv_rot = np.linalg.inv(T_icp[:3,:3]) # T 的逆的旋转部分
            t_final = source_centroid_original - T_icp_inv_rot @ target_centroid_original # 最终平移部分

            estimated_pose = np.identity(4)
            estimated_pose[:3, :3] = T_icp_inv_rot
            estimated_pose[:3, 3] = t_final

            print("\nCalculated Final Pose Matrix (Target_original -> Source_original):")
            print(estimated_pose)

            estimated_poses[inst_id] = estimated_pose # Store the final pose

            # --- 可选: 可视化对齐结果 ---
            if args.visualize_pose:
                # 为了可视化最终姿态，我们将原始 Target_pcd_original 按照 estimated_pose 进行变换
                target_pcd_transformed_by_pose = copy.deepcopy(target_pcd_original)
                target_pcd_transformed_by_pose.transform(estimated_pose)
                # 然后可视化变换后的目标点云和原始的实例点云
                visualize_icp_alignment(
                    target_pcd_transformed_by_pose, source_pcd_original, # 显示变换后的目标 和 原始实例点
                    window_name=f"Final Pose Align - Instance {inst_id}"
                )

    else:
        print("\nNo instances found, skipping ICP pose estimation.")


    # --- (可选) 保存姿态结果 ---
    if args.save_results and estimated_poses:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        # 保存为 .npz 文件，每个实例一个 key
        pose_filename = os.path.join(args.output_dir, f"estimated_poses_sample_{args.sample_index}_{timestamp}.npz")
        try:
            np.savez(pose_filename, **{f'instance_{k}': v for k, v in estimated_poses.items()})
            print(f"\nSaved estimated poses for {len(estimated_poses)} instances to {pose_filename}")
        except Exception as e_save:
            print(f"\nError saving poses: {e_save}")


    print("\nPose estimation script finished.")


# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Instance Segmentation (Semantic+Clustering) and ICP Pose Estimation on HDF5 sample. All ICP parameters are configured via command-line arguments.')

    # --- 输入/输出 ---
    parser.add_argument('--input_h5', type=str, default='./data/my_custom_dataset_h5_rgb/test_0.h5',
                        help='输入 HDF5 文件路径。')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='要处理的 HDF5 文件中的样本索引。')
    parser.add_argument('--checkpoint', type=str, default="checkpoints_seg_pyg_ptconv_rgb/best_model.pth",
                        help='预训练语义分割模型检查点 (.pth) 文件路径。')
    parser.add_argument('--model_file', type=str, default="stp/cube.STL", help='Path to the target 3D model file (e.g., screw_model.ply/stl/obj) for ICP.')
    parser.add_argument('--output_dir', type=str, default='./pose_estimation_results_args_icp', # 修改默认输出目录名称
                        help='保存输出文件的目录 (如果使用 --save_results)。')

    # --- 数据处理 ---
    parser.add_argument('--num_points', type=int, default=2048,
                        help='期望的点数 (当前代码使用 HDF5 中的实际点数)。')
    parser.add_argument('--no_rgb', action='store_true', help="Do not load/use 'rgb' data for model input (model expects 3D).")
    parser.add_argument('--normalize', action='store_true',
                        help='对点云 XYZ 坐标进行归一化用于模型输入 (中心化并缩放到单位球)。注意：这仅影响模型输入，不改变用于 ICP 的原始点坐标。')


    # --- 语义目标点选取 & DBSCAN 参数 ---
    parser.add_argument('--screw_label_id', type=int, default=1,
                        help='要进行 DBSCAN 聚类的目标语义标签 ID。')
    parser.add_argument('--dbscan_eps', type=float, default=50,
                        help='DBSCAN 的 eps 参数 (邻域半径)。需要根据点云密度和归一化后的尺度调整。')
    parser.add_argument('--dbscan_min_samples', type=int, default=1000,
                        help='DBSCAN 的 min_samples 参数 (形成核心点的最小邻居数)。需要根据预期实例大小调整。')


    # --- ICP 参数 ---
    # 将 ICP 参数全部移到这里，并提供中文帮助
    parser.add_argument('--icp_threshold', type=float, default=20,
                        help='[ICP参数] 对应点对的最大距离阈值。只有距离小于此值的源点和目标点才会被视为有效对应点。单位应与你的点云坐标单位一致。需要根据点云的噪声水平、分辨率以及预期的对齐精度进行调整。')
    parser.add_argument('--icp_estimation_method', type=str, default='point_to_point', choices=['point_to_point', 'point_to_plane'],
                        help="[ICP参数] 变换估计方法。'point_to_point': 点对点方法，最小化对应点间距离平方和，不需要法线。'point_to_plane': 点对面方法，最小化源点到目标点切平面距离平方和，需要准确法线。根据点云特性选择。")
    parser.add_argument('--icp_relative_fitness', type=float, default=1e-7,
                        help='[ICP参数] 收敛标准：Fitness 相对变化阈值。当连续两次迭代 Fitness 相对变化小于此值时停止。')
    parser.add_argument('--icp_relative_rmse', type=float, default=1e-7,
                        help='[ICP参数] 收敛标准：RMSE 相对变化阈值。当连续两次迭代 RMSE 相对变化小于此值时停止。')
    parser.add_argument('--icp_max_iter', type=int, default=1000,
                        help='[ICP参数] 收敛标准：最大迭代次数。达到此次数后强制停止。防止无限循环。')
    parser.add_argument('--icp_min_points', type=int, default=100,
                        help='[ICP参数] 进行 ICP 所需的实例点云的最小点数。点数过少可能导致 ICP 不稳定或失败。')
    # Note: ICP 的初始变换矩阵 (init) 默认设置为单位矩阵，适用于点云已中心化的情况。
    # 如果需要更复杂的初始变换，可能需要添加额外的参数或从文件加载。


    # --- 控制参数 ---
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA.')
    parser.add_argument('--save_results', action='store_true', help='Save the estimated pose matrices (.npz file).')
    parser.add_argument('--visualize_pose', action='store_true', default=True,
                        help='Visualize the final ICP alignment result for each instance.')
    # --- 模型参数 ---
    parser.add_argument('--num_classes', type=int, default=2,
                        help='模型训练时的语义类别数量。')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='(模型架构) 初始嵌入维度。')
    parser.add_argument('--k_neighbors', type=int, default=16,
                        help='(模型架构) k-NN 图中的 k 值。')
    parser.add_argument('--pt_hidden_dim', type=int, default=128,
                        help='(模型架构) Point Transformer 的隐藏层维度。')
    parser.add_argument('--pt_heads', type=int, default=4,
                        help='(模型架构) 注意力头数 (PyG PointTransformerConv 可能不直接使用)。')
    parser.add_argument('--num_transformer_layers', type=int, default=2,
                        help='(模型架构) Transformer 层的数量。')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='(模型架构) Dropout 比率。')

    args = parser.parse_args()

    # --- 依赖检查 ---
    if not SKLEARN_AVAILABLE: print("FATAL: scikit-learn required."); sys.exit(1)
    if not OPEN3D_AVAILABLE: print("FATAL: Open3D required."); sys.exit(1)

    # --- 推断模型输入维度 ---
    model_input_dim_expected = 3 if args.no_rgb else 6
    print(f"Script configured for {model_input_dim_expected}D model input features.")

    main(args)
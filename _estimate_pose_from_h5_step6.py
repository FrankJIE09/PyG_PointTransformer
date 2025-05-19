# estimate_pose_from_h5_complete_pipeline_fix_icp_call_v2.py
# 加载 HDF5 样本，执行语义分割+聚类，然后对每个实例进行 全局配准 + ICP 姿态估计。
# 强化 ICP 参数的类型转换。

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
# 确保你的 model.py 文件在同一个目录下或 sys.path 中
try:
    from model import PyG_PointTransformerSegModel # 语义分割模型类
except ImportError as e:
    print(f"FATAL Error importing model: {e}")
    print("Please ensure model.py is in the current directory or sys.path.")
    sys.exit(1)

# --- 导入 Open3D 和 Scikit-learn ---
try:
    import open3d as o3d
    # 导入配准相关模块
    from open3d.pipelines import registration as o3d_reg
    OPEN3D_AVAILABLE = True
except ImportError:
    print("FATAL Error: Open3D not found. Please install it: pip install open3d")
    sys.exit(1)
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    print("FATAL Error: scikit-learn not found. Please install it: pip install scikit-learn")
    sys.exit(1)


# --- 预处理函数 ---
def preprocess_h5_data(h5_file_path, sample_index, num_target_points, no_rgb, normalize):
    """
    加载 HDF5 数据，提取点云和 RGB (如果需要)，并可选地进行归一化。

    Args:
        h5_file_path (str): HDF5 文件路径。
        sample_index (int): 要加载的样本索引。
        num_target_points (int): 期望的点数（当前版本使用 HDF5 中的实际点数）。
        no_rgb (bool): 是否不加载/使用 RGB 数据。
        normalize (bool): 是否对点云 XYZ 坐标进行归一化用于模型输入。

    Returns:
        tuple: 包含 (features_tensor, points_original_np)。
               features_tensor (torch.Tensor): 模型输入特征张量 (1, N, C)。
               points_original_np (np.ndarray): 原始 XYZ 点云 NumPy 数组 (N, 3)。
               如果加载或处理失败，返回 (None, None)。
    """
    print(f"Loading data from HDF5: {h5_file_path}, Sample: {sample_index}")
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"HDF5 not found: {h5_file_path}")
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'data' not in f:
                raise KeyError("'data' key missing")
            dset_data = f['data']
            dset_rgb = f['rgb'] if not no_rgb and 'rgb' in f else None

            if not (0 <= sample_index < dset_data.shape[0]):
                raise IndexError("Sample index out of bounds.")

            points_xyz_np_original = dset_data[sample_index].astype(np.float32)
            points_rgb_np = dset_rgb[sample_index].astype(np.uint8) if dset_rgb is not None else None

            # 总是保留原始 XYZ 坐标用于后续的配准和姿态计算
            points_original_np = np.copy(points_xyz_np_original)

            # 根据 normalize 参数准备模型输入特征
            if normalize:
                # 对 XYZ 进行中心化和单位化，仅用于模型输入
                centroid = np.mean(points_xyz_np_original, axis=0)
                points_normalized = points_xyz_np_original - centroid
                max_dist = np.max(np.sqrt(np.sum(points_normalized ** 2, axis=1)))
                if max_dist > 1e-6:
                    points_normalized /= max_dist
                print("Applied normalization (center + unit sphere) for model input features.")
                features_list = [points_normalized]  # 使用归一化后的XYZ作为模型输入
            else:
                features_list = [points_xyz_np_original]  # 使用原始XYZ作为模型输入


            model_input_channels = 3
            if not no_rgb:
                model_input_channels = 6
                if points_rgb_np is not None:
                    features_list.append(points_rgb_np.astype(np.float32) / 255.0)
                else:
                    # 如果没有RGB，填充灰色 (0.5, 0.5, 0.5)
                    features_list.append(np.full((points_xyz_np_original.shape[0], 3), 0.5, dtype=np.float32))

            features_np = np.concatenate(features_list, axis=1)

            if features_np.shape[1] != model_input_channels:
                raise ValueError(f"Feature dim mismatch: Got {features_np.shape[1]}D, Expected {model_input_channels}D after processing.")
            if features_np.shape[0] != points_original_np.shape[0]:
                raise ValueError("Feature and point counts mismatch.")

            features_tensor = torch.from_numpy(features_np).float().unsqueeze(0)

            return features_tensor, points_original_np # 返回用于配准的原始坐标

    except Exception as e:
        print(f"Error loading/preprocessing HDF5: {e}")
        return None, None


# --- 可视化函数: 显示配准对齐结果 ---
def visualize_alignment(source_pcd_transformed, target_pcd, window_name="Alignment Result"):
    """可视化对齐后的源点云（实例）和目标点云（模型）。"""
    if not OPEN3D_AVAILABLE: return
    # 给目标和源点云不同颜色以区分
    target_pcd_vis = copy.deepcopy(target_pcd)
    source_pcd_transformed_vis = copy.deepcopy(source_pcd_transformed)
    target_pcd_vis.paint_uniform_color([0, 0.651, 0.929])  # 蓝色 (模型)
    source_pcd_transformed_vis.paint_uniform_color([1, 0.706, 0]) # 黄色 (实例)

    print(f"\nDisplaying Alignment Result for {window_name}...")
    print("Blue: Target Model | Yellow: Aligned Instance")
    print("(Close the window to continue...)")
    o3d.visualization.draw_geometries([source_pcd_transformed_vis, target_pcd_vis], window_name=window_name)
    print("Alignment visualization window closed.")


# --- 主函数 ---
def main(args):
    print(f"Starting Pose Estimation Pipeline at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
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

    # --- 加载语义模型 ---
    print(f"\nLoading SEMANTIC model checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model_input_channels = 3 if args.no_rgb else 6
    print(f"Initializing model for {model_input_channels}D input...")
    try:
        # Assuming PyG_PointTransformerSegModel constructor accepts args
        model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
        print("Model structure initialized.")
    except Exception as e:
        print(f"Error initializing model structure: {e}")
        return

    try:
        # weights_only=False to load optimizer state if needed, but usually not for inference
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        return

    try:
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint is just the state_dict
            model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading state dict into model: {e}")
        return

    model.eval() # Set model to evaluation mode


    # --- 加载和预处理 HDF5 数据 ---
    # preprocess_h5_data 返回原始 XYZ 坐标 points_original_np
    features_tensor, points_original_np = preprocess_h5_data(
        args.input_h5, args.sample_index, args.num_points, args.no_rgb, args.normalize
    )
    if features_tensor is None or points_original_np is None:
        print("Exiting due to preprocessing error.")
        return


    # --- 语义分割推理 ---
    print("\nPerforming semantic segmentation inference...")
    features_tensor = features_tensor.to(device)
    with torch.no_grad():
        logits = model(features_tensor)

    # Get predicted semantic labels
    pred_semantic_labels_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy() # (N,)
    print("Semantic prediction complete.")
    unique_semantic, counts_semantic = np.unique(pred_semantic_labels_np, return_counts=True)
    print(f"  Predicted semantic label distribution: {dict(zip(unique_semantic, counts_semantic))}")


    # --- DBSCAN 聚类 ---
    print("\nPerforming DBSCAN clustering...")
    target_semantic_label_id = args.screw_label_id # 目标物体（如螺丝）的语义标签 ID
    target_semantic_mask = (pred_semantic_labels_np == target_semantic_label_id)
    target_semantic_points_xyz = points_original_np[target_semantic_mask] # 提取目标语义类别的点云 (使用原始坐标)

    instance_labels_full = np.full_like(pred_semantic_labels_np, -1, dtype=np.int64) # 初始化完整的实例标签数组
    num_instances_found = 0
    instance_points_dict = {} # 存储每个实例的点云 {inst_id: points_array}

    if target_semantic_points_xyz.shape[0] >= args.dbscan_min_samples:
        print(f"  Applying DBSCAN (eps={args.dbscan_eps:.4f}, min_samples={args.dbscan_min_samples})...") # Added formatting
        db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
        # 对目标语义类别的点进行聚类
        instance_labels_semantic_points = db.fit_predict(target_semantic_points_xyz)

        # 将聚类结果映射回原始点云的索引
        # -1 表示噪声点，我们通常忽略噪声点作为实例
        instance_labels_full[target_semantic_mask] = instance_labels_semantic_points

        # 获取有效的实例 ID (排除噪声点 -1)
        unique_instances = np.unique(instance_labels_semantic_points[instance_labels_semantic_points != -1])
        unique_instances.sort() # 排序实例 ID 以便处理顺序一致

        num_instances_found = len(unique_instances)
        print(f"Clustering found {num_instances_found} potential instances with label {target_semantic_label_id}.")

        # 提取并存储每个实例的点云
        for inst_id in unique_instances:
            # 找到属于当前实例 ID 的点
            instance_mask_local = (instance_labels_semantic_points == inst_id)
            instance_points_dict[inst_id] = target_semantic_points_xyz[instance_mask_local]
            print(f"  Instance {inst_id}: {instance_points_dict[inst_id].shape[0]} points")
    else:
        print(f"Not enough points ({target_semantic_points_xyz.shape[0]}) predicted as target (label {target_semantic_label_id}) for DBSCAN (min_samples={args.dbscan_min_samples}).")


    # --- 加载目标 CAD/扫描模型 ---
    print(f"\nLoading target model file from: {args.model_file}")
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"Target model file not found: {args.model_file}")
    try:
        # 尝试作为三角网格加载
        target_mesh = o3d.io.read_triangle_mesh(args.model_file)

        target_pcd_original = None # 用于保存原始模型点云 (如果从网格采样) 或直接加载的点云
        target_centroid_original = np.zeros(3) # 用于保存原始模型中心 (在任何预处理前)

        if not target_mesh.has_vertices():
            # 如果加载网格失败或网格为空，尝试作为点云加载
            print("  Could not load as mesh or mesh is empty, attempting to load as point cloud...")
            target_pcd = o3d.io.read_point_cloud(args.model_file)
            if not target_pcd.has_points():
                raise ValueError("Target model file contains no points.")

            target_pcd_original = copy.deepcopy(target_pcd) # 保存原始点云
            target_centroid_original = target_pcd_original.get_center() # 保存原始中心 (点云的中心)

            # 如果是点云，为了点对面ICP/全局配准，可能需要计算法线
            if args.icp_estimation_method.lower() == 'point_to_plane' or not args.skip_global_registration:
                 print("  Estimating normals for target model point cloud...")
                 # 使用全局配准的法线半径参数进行估计 (或者 ICP 的阈值相关半径)
                 radius_for_normals = args.normal_radius_gr if not args.skip_global_registration else args.icp_threshold * 2 # Use GR radius or ICP threshold
                 max_nn_for_normals = args.max_nn_normals_gr if not args.skip_global_registration else 30 # Use GR max_nn or default
                 print(f"    Radius: {radius_for_normals:.4f}, MaxNN: {max_nn_for_normals}")
                 target_pcd.estimate_normals(
                     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_for_normals, max_nn=max_nn_for_normals)
                 )


        else:
            # 如果是网格，需要采样成点云用于配准
            print(f"  Loaded mesh with {len(target_mesh.vertices)} vertices. Sampling points for registration...")

            # 采样前保存原始中心 (使用 BBox 中心作为近似原始中心)
            mesh_bbox = target_mesh.get_axis_aligned_bounding_box()
            target_centroid_original = mesh_bbox.get_center()

            # 采样点数可以调整，例如与总输入点云数量相当，或者更多
            # 确保采样点数合理，不要太多或太少
            num_model_points_to_sample = max(args.num_points * 2, 10000) # 至少采样 10000 点，最好多于输入点云
            print(f"  Sampling {num_model_points_to_sample} points uniformly from mesh...")
            target_pcd = target_mesh.sample_points_uniformly(number_of_points=num_model_points_to_sample)

            # 注意：采样后点云的中心可能与原始网格中心略有偏差，但通常用于 ICP 影响不大
            target_pcd_original = copy.deepcopy(target_pcd) # 保存采样后的点云

            # 如果是网格采样得到的点云，为了点对面ICP/全局配准，需要计算法线
            if args.icp_estimation_method.lower() == 'point_to_plane' or not args.skip_global_registration:
                 print("  Estimating normals for sampled target model point cloud...")
                 # 使用全局配准的法线半径参数进行估计 (或者 ICP 的阈值相关半径)
                 radius_for_normals = args.normal_radius_gr if not args.skip_global_registration else args.icp_threshold * 2
                 max_nn_for_normals = args.max_nn_normals_gr if not args.skip_global_registration else 30
                 print(f"    Radius: {radius_for_normals:.4f}, MaxNN: {max_nn_for_normals}")
                 target_pcd.estimate_normals(
                     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_for_normals, max_nn=max_nn_for_normals)
                 )


        print(f"Target model loaded/sampled as point cloud with {len(target_pcd.points)} points.")
        print(f"  Original Target Model Center (estimated): [{target_centroid_original[0]:.3f},{target_centroid_original[1]:.3f},{target_centroid_original[2]:.3f}]")


        # --- 预处理目标模型: 中心化 ---
        # 将目标模型点云移到原点，便于在中心化坐标系中进行配准
        target_pcd_centered = copy.deepcopy(target_pcd)
        # 使用保存的原始中心进行平移，将模型中心移到原点
        target_pcd_centered.translate(-target_centroid_original)
        print(f"  Target model point cloud centered around origin for registration.")

    except Exception as e:
        print(f"Error loading or processing target model file: {e}")
        return


    # --- 对每个找到的实例执行 全局配准 + ICP 姿态估计 ---
    estimated_poses = {} # 存储每个实例的姿态 {inst_id: 4x4_matrix}

    if num_instances_found > 0:
        print("\nPerforming Global Registration + ICP for each found instance...")

        for inst_id, instance_points_np in instance_points_dict.items():
            print(f"\nProcessing Instance ID: {inst_id}")

            # 检查实例点数是否足够进行配准
            if instance_points_np.shape[0] < args.icp_min_points: # 使用命令行参数 icp_min_points
                 print(f"  Skipping registration: Instance has only {instance_points_np.shape[0]} points (min required: {args.icp_min_points}).")
                 estimated_poses[inst_id] = np.full((4, 4), np.nan) # 标记为未计算姿态
                 continue

            # 创建源点云 (实例)
            source_pcd_original = o3d.geometry.PointCloud()
            source_pcd_original.points = o3d.utility.Vector3dVector(instance_points_np)

            # 获取源点云的原始中心
            source_centroid_original = source_pcd_original.get_center()
            print(f"  Original Instance Point Cloud Center: [{source_centroid_original[0]:.3f},{source_centroid_original[1]:.3f},{source_centroid_original[2]:.3f}]")


            # --- 预处理源点云: 中心化 ---
            # 将实例点云移到原点，便于在中心化坐标系中进行配准
            source_pcd_centered = copy.deepcopy(source_pcd_original)
            source_pcd_centered.translate(-source_centroid_original) # 使用原始中心进行平移
            print(f"  Instance point cloud centered for registration.")

            # 如果使用点对面 ICP 或全局配准，需要计算法线
            if args.icp_estimation_method.lower() == 'point_to_plane' or not args.skip_global_registration:
                print("  Estimating normals for centered instance point cloud...")
                # 使用全局配准的法线半径参数进行估计 (或者 ICP 的阈值相关半径)
                radius_for_normals = args.normal_radius_gr if not args.skip_global_registration else args.icp_threshold * 2
                max_nn_for_normals = args.max_nn_normals_gr if not args.skip_global_registration else 30
                print(f"    Radius: {radius_for_normals:.4f}, MaxNN: {max_nn_for_normals}")
                source_pcd_centered.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_for_normals, max_nn=max_nn_for_normals)
                )
                 # 法线方向调整 (可选，但推荐对于部分扫描)
                source_pcd_centered.orient_normals_towards_camera_location() # 假设相机在原点上方


            # --- 全局配准步骤 ---
            # 在中心化后的点云上计算特征并进行全局配准，得到一个用于 ICP 的较好的初始变换
            trans_init_for_icp = np.identity(4) # 默认初始变换为单位矩阵

            if not args.skip_global_registration:
                print("\n  Performing Global Registration on CENTERED point clouds...")

                # 1. 下采样 (Downsample)
                voxel_size = args.voxel_size_gr # 全局配准的体素大小
                print(f"  Downsampling (voxel size={voxel_size:.4f})...") # Added formatting
                source_down_centered = source_pcd_centered.voxel_down_sample(voxel_size)
                target_down_centered = target_pcd_centered.voxel_down_sample(voxel_size)
                print(f"  Downsampled centered source to {len(source_down_centered.points)} points.")
                print(f"  Downsampled centered target to {len(target_down_centered.points)} points.")

                # 检查下采样后点云是否足够进行特征计算和RANSAC (至少需要ransac_n个点)
                min_points_for_gr = args.ransac_n_gr # RANSAC 每次采样的点数
                if len(source_down_centered.points) < min_points_for_gr or len(target_down_centered.points) < min_points_for_gr:
                     print(f"  Skipping global registration: Not enough downsampled points ({len(source_down_centered.points)} or {len(target_down_centered.points)}) for RANSAC (min={min_points_for_gr}).")
                     print("  Using Identity matrix as ICP initial transform.")
                     trans_init_for_icp = np.identity(4) # 点太少，无法进行全局配准，退化为单位矩阵初始化
                else:
                     # 2. 法线估计 (Estimate Normals) - FPFH 需要法线 (已在前面计算中心化点云法线时包含)
                     # 确保下采样后的点云继承了法线 (Open3D >= 0.10 应该会自动继承)
                     if not source_down_centered.has_normals() or not target_down_centered.has_normals():
                          print("  Warning: Downsampled point clouds do not have normals. Global registration might be less robust.")

                     # 3. FPFH特征计算 (Compute FPFH Features)
                     feature_radius = args.feature_radius_gr # FPFH 特征计算的搜索半径
                     max_nn_features = args.max_nn_features_gr # FPFH 最大近邻数
                     print(f"  Computing FPFH features (radius={feature_radius:.4f}, max_nn={max_nn_features})...") # Added formatting
                     source_features_centered = o3d.pipelines.registration.compute_fpfh_feature(
                         source_down_centered, o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=max_nn_features)
                     )
                     target_features_centered = o3d.pipelines.registration.compute_fpfh_feature(
                         target_down_centered, o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=max_nn_features)
                     )

                     # 4. 全局配准 (RANSAC based on Feature Matching)
                     # 使用 RANSAC 基于特征匹配寻找初始变换
                     distance_threshold_gr = args.distance_threshold_gr # 特征匹配后 RANSAC 的匹配距离阈值
                     print(f"  Running RANSAC Global Registration (distance_threshold={distance_threshold_gr:.4f})...") # Added formatting

                     # RANSAC 参数
                     ransac_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
                         max_iteration=args.ransac_max_iter_gr, # RANSAC 最大迭代次数
                         confidence=args.ransac_confidence_gr       # 信心水平
                     )
                     # RANSAC 内部估计方法 (通常是点对点)
                     ransac_estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(False) # False表示不使用法线

                     # RANSAC 每次采样的点数 (ransac_n)
                     ransac_n_points = args.ransac_n_gr # 通常是3或4

                     # Fix: Pass arguments using keywords to avoid order issues
                     global_reg_result_centered = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                         source=source_down_centered,
                         target=target_down_centered,
                         source_feature=source_features_centered,
                         target_feature=target_features_centered,
                         max_correspondence_distance=distance_threshold_gr, # <<--- Max distance for a feature match to be an inlier for RANSAC
                         estimation_method=ransac_estimation_method,
                         ransac_n=ransac_n_points,       # <<--- RANSAC 每次采样的点数
                         criteria=ransac_criteria, # RANSAC 收敛标准
                         mutual_filter=True # 通常设置为True，只考虑互为最近邻的特征匹配
                         # checkers=[] # Optional, can omit
                     )

                     # 获取全局配准得到的初始变换，作为 ICP 的 init
                     trans_init_for_icp = global_reg_result_centered.transformation

                     print("  Global Registration Result (Centered):")
                     print(f"    Fitness: {global_reg_result_centered.fitness:.4f}")
                     print(f"    Inlier RMSE: {global_reg_result_centered.inlier_rmse:.4f}")
                     print("    Estimated Initial Transformation Matrix (Global Reg, Centered):")
                     print(trans_init_for_icp)

                     # 可选：如果全局配准 Fitness 太低，可以认为全局配准失败，退化为单位矩阵初始化或标记失败
                     if global_reg_result_centered.fitness < args.min_global_fitness:
                          print(f"  Global registration fitness ({global_reg_result_centered.fitness:.4f}) is below minimum threshold ({args.min_global_fitness:.4f}). Using Identity matrix for ICP initialization.") # Added formatting
                          trans_init_for_icp = np.identity(4) # 全局配准效果差，不使用其结果
                          # 或者选择在这里就标记为失败并跳过 ICP:
                          # estimated_poses[inst_id] = np.full((4, 4), np.nan)
                          # continue # 跳过后续的ICP

            else:
                print("\n  Skipping Global Registration (due to --skip_global_registration flag). Using Identity matrix as ICP initial transform.")
                trans_init_for_icp = np.identity(4) # 不执行全局配准时，ICP初始变换为单位矩阵


            # --- ICP 精细配准步骤 ---
            print("\n  Performing ICP Refinement...")

            # 配置 ICP 参数 (从命令行参数获取)
            threshold_icp = args.icp_threshold # 对应点对的最大距离阈值 (ICP 使用自己的阈值)

            # 变换估计方法 (ICP)
            estimation_method_icp = o3d_reg.TransformationEstimationPointToPoint()
            if args.icp_estimation_method.lower() == 'point_to_plane':
                 estimation_method_icp = o3d_reg.TransformationEstimationPointToPlane()
                 print("  Warning: Using Point-to-Plane ICP. Ensure point clouds have accurate normals for ICP.")
                 # 注意：如果这里使用点对面，必须确保 source_pcd_centered 和 target_pcd_centered 有法线
                 # 法线计算已经在全局配准部分完成，确保它被正确继承或在此处重新计算

            # 收敛标准 (ICP)
            criteria_icp = o3d_reg.ICPConvergenceCriteria(
                 relative_fitness=args.icp_relative_fitness,
                 relative_rmse=args.icp_relative_rmse,
                 max_iteration=args.icp_max_iter
            )

            # ICP迭代计数器
            iteration_count_icp = [0]
            def icp_iteration_counter_callback(iteration, current_transformation, estimation_method):
                """在每次ICP迭代后被调用的回调函数"""
                iteration_count_icp[0] = iteration + 1 # 迭代次数从0开始计数


            # --- 显式转换参数类型 ---
            # 根据错误信息，确保 init 矩阵是 float64 类型，threshold 是 float 类型
            threshold_icp_float = float(threshold_icp)
            # 使用 np.asarray 增加鲁棒性，确保输入是一个数组，再进行类型转换
            trans_init_for_icp_float64 = np.asarray(trans_init_for_icp).astype(np.float64)

            print(f"  Initial ICP transform dtype: {trans_init_for_icp_float64.dtype}, shape: {trans_init_for_icp_float64.shape}")
            print(f"  ICP threshold type: {type(threshold_icp_float)}")


            # 执行 ICP
            print(f"  Running ICP (threshold={threshold_icp_float:.4f}, max_iter={criteria_icp.max_iteration})...") # Added formatting
            reg_result_icp = o3d_reg.registration_icp(
                source_pcd_centered,     # 已中心化的源点云 (实例)
                target_pcd_centered,     # 已中心化的目标点云 (模型)
                threshold_icp_float,     # ICP 对应点对最大距离阈值, 确保 float
                trans_init_for_icp_float64, # <<--- 使用初始变换, 确保 float64
                estimation_method_icp,   # ICP 变换估计方法
                criteria_icp,            # ICP 收敛标准
                callback=icp_iteration_counter_callback # 添加回调函数来统计迭代次数
            )

            # reg_result_icp 包含 fitness, inlier_rmse, transformation (这是 ICP 本身的变换)
            print(f"  ICP Result: Fitness={reg_result_icp.fitness:.4f}, RMSE={reg_result_icp.inlier_rmse:.4f}")
            print(f"  ICP iterations performed: {iteration_count_icp[0]}")
            print("  Estimated Transformation Matrix (ICP Refinement):")
            print(reg_result_icp.transformation)

            # 可选：如果 ICP Fitness 太低，可以认为配准失败
            if reg_result_icp.fitness < args.min_icp_fitness:
                 print(f"  ICP fitness ({reg_result_icp.fitness:.4f}) is below minimum threshold ({args.min_icp_fitness:.4f}). Considering registration failed.") # Added formatting
                 estimated_poses[inst_id] = np.full((4, 4), np.nan) # 标记为未计算姿态
                 continue


            # --- 计算最终姿态 (将原始模型变换到原始实例点云位置) ---
            # ICP 得到的 reg_result_icp.transformation (记为 T_icp_refine) 是在应用了 trans_init_for_icp 后，
            # 将 source_pcd_centered 进一步变换到 target_pcd_centered 的变换。
            # 整个从 source_pcd_centered 到 target_pcd_centered 的总变换是 T_total_centered = T_icp_refine @ trans_init_for_icp
            # 即 target_pcd_centered ≈ T_icp_refine @ trans_init_for_icp @ source_pcd_centered
            # (Target_original - C_target) ≈ T_total_centered @ (Source_original - C_source)
            # 我们要找的最终姿态 P 是将 Target_original 变换到 Source_original 的矩阵
            # 即 Source_original ≈ P @ Target_original
            # 反推 Source_original:
            # (T_total_centered)^-1 @ (Target_original - C_target) ≈ Source_original - C_source
            # (T_total_centered)^-1 @ Target_original - (T_total_centered)^-1 @ C_target ≈ Source_original - C_source
            # Source_original ≈ (T_total_centered)^-1 @ Target_original + C_source - (T_total_centered)^-1 @ C_target
            # 所以最终姿态矩阵 P 的旋转部分是 T_total_centered 的逆的旋转部分，平移部分是 C_source - (T_total_centered)_inv_rot @ C_target
            # 其中 C_source 是 source_centroid_original，C_target 是 target_centroid_original

            # 注意：这里使用原始的 trans_init_for_icp 进行复合计算，因为它本身可能不是单位矩阵
            Total_centered_transform = reg_result_icp.transformation @ trans_init_for_icp # ICP变换 叠加 全局配准初始变换
            Total_centered_transform_inv_rot = np.linalg.inv(Total_centered_transform[:3,:3]) # 总变换逆的旋转部分

            t_final = source_centroid_original - Total_centered_transform_inv_rot @ target_centroid_original # 最终平移部分

            estimated_pose = np.identity(4)
            estimated_pose[:3, :3] = Total_centered_transform_inv_rot
            estimated_pose[:3, 3] = t_final

            print("\nCalculated Final Pose Matrix (Target_original -> Source_original):")
            print(estimated_pose)

            estimated_poses[inst_id] = estimated_pose # Store the final pose


            # --- 可选: 可视化最终对齐结果 ---
            if args.visualize_pose:
                # 为了可视化最终姿态，我们将原始 Target_pcd_original 按照 estimated_pose 进行变换
                target_pcd_transformed_by_pose = copy.deepcopy(target_pcd_original)
                target_pcd_transformed_by_pose.transform(estimated_pose)
                # 然后可视化变换后的目标点云和原始的实例点云
                visualize_alignment(
                    target_pcd_transformed_by_pose, source_pcd_original, # 显示变换后的目标 和 原始实例点
                    window_name=f"Final Pose Align - Instance {inst_id}"
                )


    else:
        print("\nNo instances found with enough points for registration.")


    # --- (可选) 保存姿态结果 ---
    if args.save_results and estimated_poses:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        # 保存为 .npz 文件，每个实例一个 key
        pose_filename = os.path.join(args.output_dir, f"estimated_poses_sample_{args.sample_index}_{timestamp}.npz")
        try:
            # 过滤掉未计算出姿态 (NaN) 的实例
            poses_to_save = {k: v for k, v in estimated_poses.items() if isinstance(v, np.ndarray) and not np.all(np.isnan(v))}
            if poses_to_save: # Only save if there's at least one valid pose
                np.savez(pose_filename, **poses_to_save)
                print(f"\nSaved estimated poses for {len(poses_to_save)} instances to {pose_filename}")
            else:
                 print("\nNo valid poses to save.")
        except Exception as e_save:
            print(f"\nError saving poses: {e_save}")


    print("\nPose estimation pipeline finished.")


# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Instance Segmentation (Semantic+Clustering), Global Registration, and ICP Pose Estimation on HDF5 sample for partial-to-complete matching.')

    # --- 输入/输出 ---
    parser.add_argument('--input_h5', type=str, default='./data/my_custom_dataset_h5_rgb/test_0.h5',
                        help='输入 HDF5 文件路径。')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='要处理的 HDF5 文件中的样本索引。')
    parser.add_argument('--checkpoint', type=str, default="checkpoints_seg_pyg_ptconv_rgb/best_model.pth",
                        help='预训练语义分割模型检查点 (.pth) 文件路径。')
    parser.add_argument('--model_file', type=str, default="stp/cube.STL", help='Path to the target 3D model file (e.g., screw_model.ply/stl/obj) for registration.')
    parser.add_argument('--output_dir', type=str, default='./pose_estimation_results_full_pipeline', # 修改默认输出目录名称
                        help='保存输出文件的目录 (如果使用 --save_results)。')

    # --- 数据处理 ---
    parser.add_argument('--num_points', type=int, default=40960,
                        help='期望的点数 (当前代码使用 HDF5 中的实际点数)。这个参数也用于模型点云采样数量参考。')
    parser.add_argument('--no_rgb', action='store_true', help="Do not load/use 'rgb' data for model input (model expects 3D).")
    parser.add_argument('--normalize', action='store_true',
                        help='对点云 XYZ 坐标进行归一化用于模型输入 (中心化并缩放到单位球)。注意：这仅影响模型输入，不改变用于配准的原始点坐标。')


    # --- 语义目标点选取 & DBSCAN 参数 ---
    parser.add_argument('--screw_label_id', type=int, default=1,
                        help='要进行 DBSCAN 聚类的目标语义标签 ID。聚类将在这些点中进行。')
    parser.add_argument('--dbscan_eps', type=float, default=50,
                        help='[DBSCAN参数] eps 参数 (邻域半径)。需要根据目标语义点云的密度和尺度调整。')
    parser.add_argument('--dbscan_min_samples', type=int, default=1000,
                        help='[DBSCAN参数] min_samples 参数 (形成核心点的最小邻居数)。需要根据预期实例大小调整。聚类结果中点数少于此值的簇将被视为噪声。')


    # --- 全局配准参数 ---
    parser.add_argument('--voxel_size_gr', type=float, default=0.05, # 示例值，需要根据点云尺度调整
                        help='[全局配准参数] 下采样体素大小。影响特征计算和匹配的速度及精度。通常是目标物体尺寸的 1%-5%。')
    parser.add_argument('--normal_radius_gr', type=float, default=0.1, # 示例值，需要根据点云尺度调整
                        help='[全局配准参数] 法线估计的搜索半径。通常是体素大小的 2-4倍。')
    parser.add_argument('--max_nn_normals_gr', type=int, default=30,
                        help='[全局配准参数] 法线估计的 KDTree 最大近邻数。')
    parser.add_argument('--feature_radius_gr', type=float, default=0.25, # 示例值，需要根据点云尺度调整
                        help='[全局配准参数] FPFH 特征计算的搜索半径。通常是法线半径的 2-5倍。')
    parser.add_argument('--max_nn_features_gr', type=int, default=100,
                        help='[全局配准参数] FPFH 特征计算的 KDTree 最大近邻数。')
    parser.add_argument('--distance_threshold_gr', type=float, default=0.15, # 示例值，需要根据点云尺度调整
                        help='[全局配准参数] 特征匹配后 RANSAC 的匹配距离阈值。通常是体素大小的 1.5-3倍。')
    parser.add_argument('--ransac_max_iter_gr', type=int, default=100000,
                        help='[全局配准参数] RANSAC 的最大迭代次数。')
    # 移除了 ransac_max_validation_gr，因为它不在 RANSACConvergenceCriteria 构造函数中
    parser.add_argument('--ransac_confidence_gr', type=float, default=0.999,
                        help='[全局配准参数] RANSAC 的置信度。')
    parser.add_argument('--ransac_n_gr', type=int, default=3, # RANSAC 每次采样的点数，通常为3或4
                        help='[全局配准参数] RANSAC 每次随机采样的点对数量。')
    parser.add_argument('--min_global_fitness', type=float, default=0.01, # 示例阈值，可以根据全局配准的 Fitness 调整，判断是否成功
                        help='[全局配准参数] 最小全局配准 Fitness。如果 Fitness 低于此值，认为全局配准失败，会使用单位矩阵初始化 ICP。')


    # --- ICP 参数 ---
    parser.add_argument('--icp_threshold', type=float, default=0.01, # 示例值，通常比全局配准阈值小
                        help='[ICP参数] 对应点对的最大距离阈值。只考虑距离小于此值的源点和目标点。单位应与点云坐标单位一致。需要根据点云的噪声水平和局部对齐精度进行调整。')
    parser.add_argument('--icp_estimation_method', type=str, default='point_to_point', choices=['point_to_point', 'point_to_plane'],
                        help="[ICP参数] 变换估计方法。'point_to_point': 点对点方法，最小化对应点间距离平方和，不需要法线。'point_to_plane': 点对面方法，最小化源点到目标点切平面距离平方和，需要准确法线。根据点云特性选择。")
    parser.add_argument('--icp_relative_fitness', type=float, default=1e-6,
                        help='[ICP参数] 收敛标准：Fitness 相对变化阈值。当连续两次迭代 Fitness 相对变化小于此值时停止。')
    parser.add_argument('--icp_relative_rmse', type=float, default=1e-6,
                        help='[ICP参数] 收敛标准：RMSE 相对变化阈值。当连续两次迭代 RMSE 相对变化小于此值时停止。')
    parser.add_argument('--icp_max_iter', type=int, default=2000, # ICP迭代次数通常不需要像RANSAC那样高
                        help='[ICP参数] 收敛标准：最大迭代次数。达到此次数后强制停止。')
    parser.add_argument('--icp_min_points', type=int, default=30,
                        help='[ICP参数] 进行 ICP 所需的实例点云的最小点数。点数过少可能导致 ICP 不稳定或失败。也用于筛选聚类结果。')
    parser.add_argument('--min_icp_fitness', type=float, default=0.5, # 示例阈值
                        help='[ICP参数] 最小 ICP Fitness。如果最终 Fitness 低于此值，认为 ICP 配准失败。')

    # --- 控制参数 ---
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA.')
    parser.add_argument('--save_results', action='store_true', help='Save the estimated pose matrices (.npz file)。NaN 姿态不会被保存。')
    parser.add_argument('--visualize_pose', action='store_true', default=True,
                        help='Visualize the final alignment result for each instance.')
    parser.add_argument('--skip_global_registration', action='store_true',
                        help='跳过全局配准步骤，直接使用单位矩阵作为 ICP 初始变换 (不推荐用于局部-整体配准)。')

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
    if not SKLEARN_AVAILABLE:
        print("FATAL: scikit-learn required.")
        sys.exit(1)
    if not OPEN3D_AVAILABLE:
        print("FATAL: Open3D required.")
        sys.exit(1)

    # --- 推断模型输入维度 ---
    model_input_dim_expected = 3 if args.no_rgb else 6
    print(f"Script configured for {model_input_dim_expected}D model input features.")

    main(args)
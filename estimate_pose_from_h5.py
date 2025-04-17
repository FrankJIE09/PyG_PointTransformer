# estimate_pose_from_h5.py
# 加载 HDF5 样本，执行语义分割+聚类，然后对每个实例进行 ICP 姿态估计。

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


# --- 预处理函数 (与 segment_instances_from_h5.py 相同) ---
def preprocess_h5_data(h5_file_path, sample_index, num_target_points, no_rgb, normalize):
    # ... (代码与上一个版本完全相同 - 返回 features_tensor, points_processed_np) ...
    # 返回: 模型输入 Tensor(1,N,C), 处理后的 XYZ NumPy(N,3)
    print(f"Loading data from HDF5: {h5_file_path}, Sample: {sample_index}")
    if not os.path.exists(h5_file_path): raise FileNotFoundError(f"HDF5 not found: {h5_file_path}")
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'data' not in f: raise KeyError("'data' key missing")
            dset_data = f['data']; dset_rgb = f['rgb'] if not no_rgb and 'rgb' in f else None
            if not (0 <= sample_index < dset_data.shape[0]): raise IndexError("Sample index out of bounds.")
            points_xyz_np = dset_data[sample_index].astype(np.float32)
            points_rgb_np = dset_rgb[sample_index].astype(np.uint8) if dset_rgb is not None else None
            if points_xyz_np.shape[0] != num_target_points: print(f"Warn: HDF5 points ({points_xyz_np.shape[0]}) != target ({num_target_points}). Using HDF5 points.")
            points_processed_np = np.copy(points_xyz_np)
            if normalize:
                centroid = np.mean(points_processed_np, axis=0); points_processed_np -= centroid
                max_dist = np.max(np.sqrt(np.sum(points_processed_np ** 2, axis=1)))
                if max_dist > 1e-6: points_processed_np /= max_dist
                print("Applied normalization (center + unit sphere).")
            features_list = [points_processed_np]
            model_input_channels = 3
            if not no_rgb:
                model_input_channels = 6
                if points_rgb_np is not None: features_list.append(points_rgb_np.astype(np.float32) / 255.0)
                else: features_list.append(np.full((points_processed_np.shape[0], 3), 0.5, dtype=np.float32))
            features_np = np.concatenate(features_list, axis=1)
            if features_np.shape[1] != model_input_channels: raise ValueError(f"Feature dim mismatch: Got {features_np.shape[1]}D, Expected {model_input_channels}D")
            features_tensor = torch.from_numpy(features_np).float().unsqueeze(0)
            return features_tensor, points_processed_np.astype(np.float32)
    except Exception as e: print(f"Error loading/preprocessing HDF5: {e}"); return None, None


# --- (新增) 可视化函数: 显示 ICP 对齐结果 ---
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
    features_tensor, points_processed_np = preprocess_h5_data(
        args.input_h5, args.sample_index, args.num_points, args.no_rgb, args.normalize
    )
    if features_tensor is None: print("Exiting due to preprocessing error."); return


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
    screw_points_xyz = points_processed_np[screw_mask]
    instance_labels_full = np.full_like(pred_semantic_labels_np, -1, dtype=np.int64)
    num_instances_found = 0
    instance_points_dict = {} # 存储每个实例的点云 {inst_id: points_array}

    if screw_points_xyz.shape[0] >= args.dbscan_min_samples:
        db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
        instance_labels_screw_points = db.fit_predict(screw_points_xyz)
        instance_labels_full[screw_mask] = instance_labels_screw_points
        unique_instances = np.unique(instance_labels_screw_points[instance_labels_screw_points != -1])
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
        if not target_mesh.has_vertices(): # 如果失败或为空，尝试作为点云加载
             print("  Could not load as mesh or mesh is empty, attempting to load as point cloud...")
             target_pcd = o3d.io.read_point_cloud(args.model_file)
             if not target_pcd.has_points(): raise ValueError("Target model file contains no points.")
             # 检查是否有点，但可能需要法线用于 P2Plane ICP，这里先不计算
        else:
             # 如果是网格，需要采样成点云才能用于点对点 ICP
             # 或者计算网格顶点法线，使用 P2Plane ICP
             print(f"  Loaded mesh with {len(target_mesh.vertices)} vertices. Sampling points for ICP...")
             # 采样点数可以调整，例如与实例点云数量相当，或者更多
             num_model_points = max(args.num_points, 5000) # 至少采样 5000 点
             target_pcd = target_mesh.sample_points_uniformly(number_of_points=num_model_points)
             # 可选：计算法线
             # target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.dbscan_eps * 2, max_nn=30))

        print(f"Target model loaded as point cloud with {len(target_pcd.points)} points.")

        # --- 预处理目标模型: 中心化 ---
        # 将目标模型移到原点，ICP 对齐时更容易处理
        target_centroid = target_pcd.get_center()
        target_pcd.translate(-target_centroid)
        print(f"  Target model centered around origin.")

    except Exception as e:
        print(f"Error loading or processing target model file: {e}"); return


    # --- 对每个找到的实例执行 ICP 姿态估计 ---
    estimated_poses = {} # 存储每个实例的姿态 {inst_id: 4x4_matrix}
    if num_instances_found > 0:
        print("\nPerforming ICP for each found instance...")
        # ICP 参数设置
        threshold = args.icp_threshold # 对应点对的最大距离
        # 初始变换矩阵 (单位矩阵，因为我们假设对源和目标都进行了中心化)
        trans_init = np.identity(4)
        # ICP 收敛标准
        criteria = o3d_reg.ICPConvergenceCriteria(
             relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=args.icp_max_iter
        )
        # 评估方法 (点对点 Point-to-Point)
        estimation_method = o3d_reg.TransformationEstimationPointToPoint()
        # estimation_method = o3d_reg.TransformationEstimationPointToPlane() # 如果有法线可用

        for inst_id, instance_points_np in instance_points_dict.items():
            print(f"\nProcessing Instance ID: {inst_id}")
            if instance_points_np.shape[0] < 10: # 点太少无法进行 ICP
                 print(f"  Skipping ICP: Instance has only {instance_points_np.shape[0]} points.")
                 continue

            # 创建源点云 (实例)
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(instance_points_np)

            # --- 预处理源点云: 中心化 ---
            # 注意：这里中心化是为了 ICP 对齐，最终的变换矩阵需要考虑这个中心化偏移
            source_centroid = source_pcd.get_center()
            source_pcd.translate(-source_centroid)
            print(f"  Instance centered for ICP (Original centroid: [{source_centroid[0]:.3f},{source_centroid[1]:.3f},{source_centroid[2]:.3f}])")

            # 可选：计算法线 (如果使用 P2Plane)
            # source_pcd.estimate_normals(...)

            # 执行 ICP
            print(f"  Running ICP (Threshold={threshold}, MaxIter={args.icp_max_iter})...")
            reg_result = o3d_reg.registration_icp(
                source_pcd, target_pcd, threshold, trans_init,
                estimation_method, criteria
            )

            # reg_result 包含 fitness, inlier_rmse, transformation
            print(f"  ICP Result: Fitness={reg_result.fitness:.4f}, RMSE={reg_result.inlier_rmse:.4f}")
            print("  Estimated Transformation Matrix (Model to Instance, relative to centroids):")
            print(reg_result.transformation)

            # --- 计算最终姿态 ---
            # ICP 得到的 T_icp 是将中心化的 source_pcd 对齐到中心化的 target_pcd
            # 即: target_pcd ≈ T_icp * source_pcd_centered
            # 我们需要的是将原始 target_pcd (在原点) 变换到 原始 source_pcd (在场景中) 的位姿 T_final
            # 原始 source_pcd = source_pcd_centered + source_centroid
            # 原始 target_pcd = target_pcd_centered + target_centroid (这里 target_centroid 是 0，因为我们加载后就中心化了)
            # target_pcd ≈ T_icp * (原始 source_pcd - source_centroid)
            # T_icp^-1 * target_pcd + source_centroid ≈ 原始 source_pcd
            # 最终变换 T_final 应该是：先将 target 移到原点 (已完成)，应用 T_icp^-1，然后平移到 source_centroid
            # T_final = Translate(source_centroid) * Inverse(T_icp) * Translate(-target_centroid)
            # 因为 target_centroid 是 0, T_final = Translate(source_centroid) * Inverse(T_icp)
            # 或者更直接地，T_icp 将 source 对齐到 target, 我们要反过来
            # T_final 是将模型坐标系下的点变换到相机坐标系下该实例的位置
            # T_final = Translate(source_centroid) * T_icp * Translate(-source_centroid) ?? 不对
            # T_icp: source_centered -> target_centered
            # 我们需要: target_origin -> source_origin
            # T_target_origin_to_centered = Translate(-target_centroid)
            # T_source_centered_to_origin = Translate(source_centroid)
            # T_target_centered_to_source_centered = Inverse(T_icp)
            # T_final = T_source_centered_to_origin * T_target_centered_to_source_centered * T_target_origin_to_centered
            # T_final = Translate(source_centroid) * np.linalg.inv(reg_result.transformation) * np.identity(4) #因为 target_centroid 是 0

            # 另一种理解：transformation 是将 source 对齐到 target 的变换
            # T_target = transformation * T_source
            # 我们要找的是 T_pose 使得 T_pose * Model = Instance
            # Model ≈ T_target_centered
            # Instance ≈ T_source_origin = T_source_centered + source_centroid
            # T_target_centered ≈ transformation * T_source_centered
            # Model ≈ transformation * (Instance - source_centroid)
            # Inverse(transformation) * Model ≈ Instance - source_centroid
            # Instance ≈ Inverse(transformation) * Model + source_centroid
            # 所以 T_pose 应该是一个先应用 Inverse(transformation)，然后平移 source_centroid 的变换。
            # 这似乎也不对... 重新思考ICP输出：
            # reg_result.transformation T 使得 T * source ≈ target
            # source 是 instance 点云（已中心化到 S），target 是 model 点云（已中心化到 T=0）
            # 即 T * (Instance - S) ≈ (Model - T) = Model
            # 我们要找 T_pose 使得 T_pose * Model ≈ Instance
            # 从上式解 Instance = T^-1 * Model + S
            # 所以 T_pose = Translate(S) * Inverse(T)
            T_icp = reg_result.transformation
            T_pose = np.identity(4)
            T_pose[:3, 3] = source_centroid # 设置平移部分为实例的原始中心
            T_pose[:3, :3] = np.linalg.inv(T_icp[:3, :3]) # 设置旋转部分为 ICP 旋转的逆

            # 或者，如果 Open3D 定义是 T 把 source 移到 target 位置：
            # 那么 Model(centered) = T_icp * Instance(centered)
            # Model = T_icp * (Instance - instance_centroid)
            # T_icp^-1 * Model = Instance - instance_centroid
            # Instance = T_icp^-1 * Model + instance_centroid
            # 所以 T_pose 是先应用 T_icp^-1 再平移 instance_centroid
            T_pose_inv = np.linalg.inv(reg_result.transformation)
            T_final_pose = np.identity(4)
            T_final_pose[:3,:3] = T_pose_inv[:3,:3] # 旋转
            T_final_pose[:3, 3] = source_centroid + T_pose_inv[:3,:3] @ (-target_centroid) # 平移
            # 因为 target_centroid=0, T_final_pose[:3, 3] = source_centroid

            final_transformation = np.identity(4)
            final_transformation[:3, :3] = reg_result.transformation[:3, :3] # ICP 找到的旋转
            # 平移部分 = 实例的中心点 - ICP旋转作用于模型中心点(0) + ICP平移 (???)
            # 变换后的模型中心 = R_icp * target_centroid + t_icp = t_icp (因为 target_centroid = 0)
            # 我们希望 变换后的模型中心 = instance_centroid
            # 所以 t_final = instance_centroid
            # 但这只在模型和实例都中心化时才对...

            # 最直接的理解：reg_result.transformation 是将 source_pcd 变换到 target_pcd 坐标系的矩阵
            # 即 T aligns source to target.
            estimated_poses[inst_id] = reg_result.transformation
            print(f"  Stored transformation matrix for instance {inst_id}")

            # --- 可选: 可视化对齐结果 ---
            if args.visualize_pose:
                source_pcd_transformed = copy.deepcopy(source_pcd)
                source_pcd_transformed.transform(reg_result.transformation) # 将源点云变换到目标坐标系
                visualize_icp_alignment(
                    source_pcd_transformed, target_pcd, # 显示变换后的源 和 原始目标
                    window_name=f"ICP Align - Instance {inst_id}"
                )

    else:
        print("\nNo instances found, skipping ICP pose estimation.")


    # --- (可选) 保存姿态结果 ---
    if args.save_results and estimated_poses:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        pose_filename = os.path.join(args.output_dir, f"estimated_poses_{timestamp}.npz")
        try:
            np.savez(pose_filename, **{f'instance_{k}': v for k, v in estimated_poses.items()})
            print(f"\nSaved estimated poses for {len(estimated_poses)} instances to {pose_filename}")
        except Exception as e_save:
            print(f"\nError saving poses: {e_save}")


    print("\nPose estimation script finished.")


# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Instance Segmentation (Semantic+Clustering) and ICP Pose Estimation on HDF5 sample.')

    # --- 输入/输出 ---
    parser.add_argument('--input_h5', type=str, default='./data/my_custom_dataset_h5_rgb/test_0.h5',
                        help='输入 HDF5 文件路径。')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='要处理的 HDF5 文件中的样本索引。')
    parser.add_argument('--checkpoint', type=str, default="checkpoints_seg_pyg_ptconv_rgb/best_model.pth",
                        help='预训练模型检查点 (.pth) 文件路径。')
    parser.add_argument('--model_file', type=str,default="stp/DIN_912-M8x30.stl", help='Path to the target 3D model file (e.g., screw_model.ply/stl/obj) for ICP.') # 新增
    parser.add_argument('--output_dir', type=str, default='./segmentation_results_h5_no_try',
                        help='保存输出文件的目录 (如果使用 --save_results)。')
    # --- 数据处理 ---
    parser.add_argument('--num_points', type=int, default=40960,
                        help='期望的点数 (仅供参考, 代码会使用 HDF5 中的实际点数)。')
    parser.add_argument('--no_rgb', action='store_true', help="Do not load/use 'rgb' data (model expects 3D).")
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='对点云 XYZ 坐标进行归一化 (中心化并缩放到单位球)。(默认: True)')
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


    # --- 聚类参数 ---
    # --do_clustering is implicitly True
    parser.add_argument('--screw_label_id', type=int, default=0,
                        help='要进行 DBSCAN 聚类的目标语义标签 ID。')
    parser.add_argument('--dbscan_eps', type=float, default=20,
                        help='DBSCAN 的 eps 参数 (邻域半径)。需要根据点云密度和归一化后的尺度调整。')
    parser.add_argument('--dbscan_min_samples', type=int, default=10,
                        help='DBSCAN 的 min_samples 参数 (形成核心点的最小邻居数)。需要根据预期实例大小调整。')

    # --- ICP 参数 ---
    parser.add_argument('--icp_threshold', type=float, default=10, help='ICP correspondence distance threshold (meters, default: 0.01 for 1cm). TUNABLE.') # 新增
    parser.add_argument('--icp_max_iter', type=int, default=100, help='ICP maximum iterations (default: 100).') # 新增

    # --- 控制参数 ---
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA.')
    parser.add_argument('--save_results', action='store_true', help='Save results (NPY, TXT, Instance PLYs, Poses).')
    parser.add_argument('--visualize_pose', action='store_true', default=True,help='Visualize ICP alignment result for each instance.') # 新增可视化开关

    args = parser.parse_args()

    # --- 依赖检查 ---
    if not SKLEARN_AVAILABLE: print("FATAL: scikit-learn required."); sys.exit(1)
    if not OPEN3D_AVAILABLE: print("FATAL: Open3D required."); sys.exit(1)

    # --- 推断模型输入维度 ---
    model_input_dim_expected = 3 if args.no_rgb else 6
    print(f"Script configured for {model_input_dim_expected}D model input features.")

    main(args)
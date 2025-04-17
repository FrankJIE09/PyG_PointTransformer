# estimate_pose_from_h5.py
# 版本: 语义分割+聚类+ICP姿态估计，最终可视化场景中放置的模型

import torch
import numpy as np
import argparse
import time
import os
import datetime
import sys
import h5py
import copy

# --- 导入本地模块 ---
try:
    from model import PyG_PointTransformerSegModel
except ImportError as e: print(f"FATAL Error importing model: {e}"); sys.exit(1)

# --- 导入 Open3D 和 Scikit-learn ---
try:
    import open3d as o3d
    from open3d.pipelines import registration as o3d_reg
    OPEN3D_AVAILABLE = True
except ImportError: print("FATAL Error: Open3D not found."); sys.exit(1)
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError: print("FATAL Error: scikit-learn not found."); sys.exit(1)


# --- 预处理函数 (不变) ---
def preprocess_h5_data(h5_file_path, sample_index, num_target_points, no_rgb, normalize):
    # ... (代码与上一个版本完全相同) ...
    # 返回: 模型输入 Tensor(1,N,C), 处理后的 XYZ NumPy(N,3), 原始 XYZ NumPy(N,3)
    print(f"加载HDF5数据: {h5_file_path}, 样本: {sample_index}")
    if not os.path.exists(h5_file_path): raise FileNotFoundError(f"HDF5未找到: {h5_file_path}")
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'data' not in f: raise KeyError("'data' 键缺失")
            dset_data = f['data']; dset_rgb = f['rgb'] if not no_rgb and 'rgb' in f else None
            if not (0 <= sample_index < dset_data.shape[0]): raise IndexError("样本索引越界")
            points_xyz_np_orig = dset_data[sample_index].astype(np.float32) # 原始XYZ (采样/填充后)
            points_rgb_np = dset_rgb[sample_index].astype(np.uint8) if dset_rgb is not None else None
            num_points = points_xyz_np_orig.shape[0]
            if num_points != num_target_points: print(f"信息: HDF5点数({num_points}) != 目标({num_target_points})。使用实际点数。")
            points_processed_np = np.copy(points_xyz_np_orig) # 用于处理和模型输入
            if normalize:
                centroid = np.mean(points_processed_np, axis=0); points_processed_np -= centroid
                max_dist = np.max(np.sqrt(np.sum(points_processed_np ** 2, axis=1)));
                if max_dist > 1e-6: points_processed_np /= max_dist
                print("已应用坐标归一化。")
            features_list = [points_processed_np]
            model_input_channels = 3
            if not no_rgb:
                model_input_channels = 6
                if points_rgb_np is not None: features_list.append(points_rgb_np.astype(np.float32) / 255.0)
                else: features_list.append(np.full((points_processed_np.shape[0], 3), 0.5, dtype=np.float32))
            features_np = np.concatenate(features_list, axis=1)
            if features_np.shape[1] != model_input_channels: raise ValueError("特征维度不匹配")
            features_tensor = torch.from_numpy(features_np).float().unsqueeze(0)
            # 返回模型输入, 处理后的XYZ坐标(用于聚类), 原始XYZ坐标(用于参考最终位姿)
            return features_tensor, points_processed_np.astype(np.float32), points_xyz_np_orig
    except Exception as e: print(f"加载/预处理HDF5时出错: {e}"); return None, None, None


# --- (新增) 可视化函数: 显示场景和带有估计姿态的模型 ---
def visualize_scene_with_poses(scene_points_np, scene_semantic_labels, estimated_final_poses, target_model_orig, num_classes, screw_label_id, window_name="Pose Estimation Result"):
    """
    可视化原始场景点云（背景灰色）以及根据估计姿态放置的标准模型。

    Args:
        scene_points_np (np.ndarray): 原始场景处理后的 XYZ 点 (N, 3)。
        scene_semantic_labels (np.ndarray): 场景点的语义标签 (N,)。
        estimated_final_poses (dict): {instance_id: final_pose_matrix (4x4)}。
        target_model_orig (o3d.geometry.TriangleMesh or o3d.geometry.PointCloud): 原始标准模型。
        num_classes (int): 语义类别总数 (用于颜色映射)。
        screw_label_id (int): 螺丝钉的语义标签 ID。
        window_name (str): 窗口标题。
    """
    if not OPEN3D_AVAILABLE: print("Open3D not available."); return
    if scene_points_np is None or scene_semantic_labels is None: return

    geometries = []

    # 1. 创建场景背景点云 (非螺丝钉部分，灰色)
    background_mask = (scene_semantic_labels != screw_label_id)
    background_points = scene_points_np[background_mask]
    if background_points.shape[0] > 0:
        pcd_background = o3d.geometry.PointCloud()
        pcd_background.points = o3d.utility.Vector3dVector(background_points)
        pcd_background.paint_uniform_color([0.7, 0.7, 0.7]) # 灰色背景
        geometries.append(pcd_background)
    else:
         print("警告: 未找到背景点用于显示。")

    # 2. 放置估计姿态的模型
    num_instances = len(estimated_final_poses)
    if num_instances > 0:
        print(f"  正在可视化 {num_instances} 个估计姿态的模型...") # 中文
        # 为实例生成颜色
        np.random.seed(42)
        instance_colors = np.random.rand(num_instances, 3)
        instance_ids = sorted(estimated_final_poses.keys())

        for i, inst_id in enumerate(instance_ids):
            final_pose = estimated_final_poses[inst_id]
            # 复制原始模型并应用最终姿态变换
            # 检查 target_model_orig 是 Mesh 还是 PointCloud
            if isinstance(target_model_orig, o3d.geometry.TriangleMesh):
                 model_instance = copy.deepcopy(target_model_orig) # 复制网格
                 if not model_instance.has_vertex_normals(): model_instance.compute_vertex_normals() # 计算法线
            elif isinstance(target_model_orig, o3d.geometry.PointCloud):
                 model_instance = copy.deepcopy(target_model_orig) # 复制点云
            else:
                 print(f"错误: 不支持的目标模型类型 {type(target_model_orig)}") # 中文
                 continue

            model_instance.transform(final_pose) # 应用计算出的最终变换
            model_instance.paint_uniform_color(instance_colors[i]) # 上色
            geometries.append(model_instance)
            # print(f"  实例 {inst_id} 已放置。") # 中文

    else:
        print("  未找到实例来放置模型。") # 中文
        # 如果没有实例，可能只显示背景，或者显示语义分割结果？
        # 为了保持一致，这里只显示背景（如果有的话）

    # 3. 显示
    if not geometries:
         print("警告: 没有可供显示的几何体。") # 中文
         return

    print(f"\n正在显示包含已定位模型的场景 ({num_instances} 个实例)...") # 中文
    print("(请关闭 Open3D 窗口以完成脚本)") # 中文
    o3d.visualization.draw_geometries(geometries, window_name=window_name, width=1024, height=768)
    print("姿态可视化窗口已关闭。") # 中文


# --- 主函数 ---
def main(args):
    print(f"脚本启动于: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}") # 中文
    print(f"参数: {args}")
    if not SKLEARN_AVAILABLE or not OPEN3D_AVAILABLE: print("致命错误: 需要scikit-learn和Open3D。"); sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"); print(f"使用设备: {device}")
    if args.save_results: os.makedirs(args.output_dir, exist_ok=True)

    # --- 加载语义模型 ---
    print(f"\n加载语义模型检查点: {args.checkpoint}")
    if not os.path.exists(args.checkpoint): raise FileNotFoundError(f"检查点未找到: {args.checkpoint}")
    model_input_channels = 3 if args.no_rgb else 6; print(f"初始化模型 ({model_input_channels}D 输入)...")
    try: model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device); print("模型结构已初始化。")
    except Exception as e: print(f"初始化模型错误: {e}"); return
    try: checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except Exception as e: print(f"加载检查点文件错误: {e}"); return
    try:
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.load_state_dict(checkpoint)
        print("模型权重加载成功。")
    except Exception as e: print(f"加载状态字典错误: {e}"); return
    model.eval()

    # --- 加载和预处理 HDF5 数据 ---
    features_tensor, points_processed_np, points_original_xyz_np = preprocess_h5_data(
        args.input_h5, args.sample_index, args.num_points, args.no_rgb, args.normalize
    )
    if features_tensor is None: print("预处理错误，退出。"); return
    # points_processed_np 是可能归一化后的坐标 (N, 3)，用于聚类
    # points_original_xyz_np 是 HDF5 中的原始坐标 (N, 3)，用于参考位姿? 可能不需要了
    # 我们需要的是未经归一化的，但经过采样/填充后的点云坐标来显示场景
    # 假设 points_processed_np 如果没归一化就是我们想要的场景坐标；如果归一化了，需要反归一化才能显示原始场景
    # 为了简化，假设 points_processed_np (即使归一化了) 作为可视化场景坐标，最终位姿也是基于此计算
    scene_points_for_viz = points_processed_np

    # --- 语义分割推理 ---
    print("\n执行语义分割推理...")
    features_tensor = features_tensor.to(device)
    with torch.no_grad(): logits = model(features_tensor)
    pred_semantic_labels_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
    print("语义预测完成。")
    unique_semantic, counts_semantic = np.unique(pred_semantic_labels_np, return_counts=True)
    print(f"  预测语义标签分布: {dict(zip(unique_semantic, counts_semantic))}")

    # --- DBSCAN 聚类 ---
    print("\n执行 DBSCAN 聚类...")
    screw_label_id = args.screw_label_id
    screw_mask = (pred_semantic_labels_np == screw_label_id)
    screw_points_xyz = points_processed_np[screw_mask] # 使用处理/归一化后的坐标聚类
    instance_labels_full = np.full_like(pred_semantic_labels_np, -1, dtype=np.int64)
    num_instances_found = 0
    instance_points_dict = {} # 存储实例点 (处理后坐标)
    instance_centroids_dict = {} # 存储实例中心 (处理后坐标)

    if screw_points_xyz.shape[0] >= args.dbscan_min_samples:
        db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
        instance_labels_screw_points = db.fit_predict(screw_points_xyz)
        instance_labels_full[screw_mask] = instance_labels_screw_points
        unique_instances = np.unique(instance_labels_screw_points[instance_labels_screw_points != -1])
        num_instances_found = len(unique_instances)
        print(f"聚类找到 {num_instances_found} 个潜在实例。")
        for inst_id in unique_instances:
            instance_mask_local = (instance_labels_screw_points == inst_id)
            instance_points = screw_points_xyz[instance_mask_local]
            if instance_points.shape[0] > 0:
                instance_points_dict[inst_id] = instance_points
                instance_centroids_dict[inst_id] = np.mean(instance_points, axis=0) # 质心
                print(f"  实例 {inst_id}: {instance_points.shape[0]} 个点, 质心=[{instance_centroids_dict[inst_id][0]:.3f},{instance_centroids_dict[inst_id][1]:.3f},{instance_centroids_dict[inst_id][2]:.3f}]")
    else:
        print(f"未找到足够的目标点进行 DBSCAN。")

    # --- 加载目标模型 (原始的，用于可视化) ---
    print(f"\n加载用于可视化的目标模型文件: {args.model_file}")
    if not os.path.exists(args.model_file): raise FileNotFoundError(f"目标模型文件未找到: {args.model_file}")
    try:
        target_model_orig = o3d.io.read_triangle_mesh(args.model_file) # 尝试加载网格
        if not target_model_orig.has_vertices(): # 如果失败，尝试加载点云
             target_model_orig = o3d.io.read_point_cloud(args.model_file)
             if not target_model_orig.has_points(): raise ValueError("目标模型文件无效")
             print("目标模型作为点云加载。")
        else: print("目标模型作为网格加载。")
        # 计算原始模型的质心，用于后续姿态计算
        target_centroid_orig = target_model_orig.get_center()
        print(f"  目标模型的原始质心: {target_centroid_orig}")
    except Exception as e: print(f"加载或处理目标模型文件错误: {e}"); return


    # --- 对每个找到的实例执行 ICP 并计算最终姿态 ---
    estimated_final_poses = {} # 存储最终姿态 {inst_id: T_final_pose (4x4)}
    if num_instances_found > 0:
        print("\n对每个找到的实例执行 ICP 姿态估计...")

        # --- 准备用于 ICP 的目标点云 (采样+中心化) ---
        try:
            target_pcd_icp = copy.deepcopy(target_model_orig)
            if isinstance(target_pcd_icp, o3d.geometry.TriangleMesh):
                 num_model_points = max(args.num_points // 10, 2048) # 采样点数调整
                 print(f"  从目标网格采样 {num_model_points} 点用于 ICP...")
                 target_pcd_icp = target_pcd_icp.sample_points_uniformly(number_of_points=num_model_points)
            # 计算 ICP 目标的中心点 (加载时的原始模型)
            target_centroid_icp = target_pcd_icp.get_center() # 这个中心点 T
            target_pcd_icp.translate(-target_centroid_icp) # 中心化 T_c = M - T
            print("  用于 ICP 的目标点云已准备并中心化。")
        except Exception as e: print(f"准备 ICP 目标点云时出错: {e}"); return

        # --- ICP 参数 ---
        threshold = args.icp_threshold; trans_init = np.identity(4)
        criteria = o3d_reg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=args.icp_max_iter)
        estimation_method = o3d_reg.TransformationEstimationPointToPoint()

        for inst_id, instance_points_np in instance_points_dict.items():
            print(f"\n处理实例 ID: {inst_id}")
            if instance_points_np.shape[0] < 10: print(f"  跳过 ICP: 点数不足。"); continue

            source_pcd = o3d.geometry.PointCloud(); source_pcd.points = o3d.utility.Vector3dVector(instance_points_np)
            source_centroid = np.mean(instance_points_np, axis=0) # 实例的中心点 S (在处理后的坐标系中)
            source_pcd_centered = copy.deepcopy(source_pcd).translate(-source_centroid) # 中心化实例 S_c = I - S
            print(f"  实例已中心化用于 ICP (质心 S = [{source_centroid[0]:.3f},{source_centroid[1]:.3f},{source_centroid[2]:.3f}])")

            print(f"  运行 ICP...")
            reg_result = o3d_reg.registration_icp(source_pcd_centered, target_pcd_icp, threshold, trans_init, estimation_method, criteria)
            T_icp = reg_result.transformation # T_icp: S_c -> T_c (M-T)

            # --- 计算最终姿态 T_final_pose: M_orig -> I_orig (近似为 I_processed) ---
            # T_final = Translate(S) * Inverse(T_icp) * Translate(-T)
            T_icp_inv = np.linalg.inv(T_icp)
            T_final_pose = np.identity(4)
            R_final = T_icp_inv[:3,:3]
            t_final = source_centroid - R_final @ target_centroid_icp # T 是 ICP 用的目标模型的中心点
            T_final_pose[:3,:3] = R_final
            T_final_pose[:3,3] = t_final

            estimated_final_poses[inst_id] = T_final_pose
            print(f"  ICP 结果: Fitness={reg_result.fitness:.4f}, RMSE={reg_result.inlier_rmse:.4f}")
            print(f"  计算得到的最终姿态矩阵 (用于放置模型) 已存储。")

    else:
        print("\n未找到实例，跳过 ICP 姿态估计。")

    # --- (可选) 保存结果 ---
    if args.save_results:
        # ... (保存 NPY, TXT 逻辑不变) ...
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        output_dir = args.output_dir; os.makedirs(output_dir, exist_ok=True)
        print(f"\n保存结果... 时间戳: {timestamp}")
        np.save(os.path.join(output_dir, f"points_processed_{timestamp}.npy"), points_processed_np)
        np.save(os.path.join(output_dir, f"semantic_labels_{timestamp}.npy"), pred_semantic_labels_np)
        np.save(os.path.join(output_dir, f"instance_labels_{timestamp}.npy"), instance_labels_full)
        data_to_save = np.hstack((points_processed_np, pred_semantic_labels_np.reshape(-1, 1), instance_labels_full.reshape(-1, 1)))
        txt_filename = os.path.join(output_dir, f"segmented_full_{timestamp}.txt"); np.savetxt(txt_filename, data_to_save, fmt='%.6f %.6f %.6f %d %d', delimiter=' ')
        # 保存姿态矩阵
        if estimated_final_poses:
             pose_filename = os.path.join(output_dir, f"estimated_final_poses_{timestamp}.npz")
             try: np.savez(pose_filename, **{f'instance_{k}': v for k, v in estimated_final_poses.items()}); print(f"已保存姿态到 {pose_filename}")
             except Exception as e_save: print(f"保存姿态错误: {e_save}")
        print("结果保存完成。")

    # --- (修改) 最终可视化: 显示带有姿态模型的场景 ---
    if not args.no_visualize and OPEN3D_AVAILABLE:
        visualize_scene_with_poses(
            points_processed_np,      # 场景点云 (处理后的 XYZ)
            pred_semantic_labels_np,  # 语义标签 (用于区分背景)
            estimated_final_poses,    # 最终姿态字典
            target_model_orig,        # 原始标准模型 (网格或点云)
            args.num_classes,
            args.screw_label_id,
            window_name=f"Scene with Posed Models - Sample {args.sample_index}"
        )
    else:
        print("\n跳过最终可视化。")

    # --- 脚本结束 ---
    # end_time = datetime.datetime.now(); print(f"\n脚本完成于: {end_time.strftime('%Y-%m-%d_%H-%M-%S')}"); print(f"总执行时间: {end_time - start_time}")


# --- 命令行参数解析 (移除 visualize_pose, marker_radius) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='实例分割与 ICP 姿态估计 (HDF5 输入)，可视化最终场景。') # 中文

    # 输入/输出
    parser.add_argument('--input_h5', type=str, default='./data/my_custom_dataset_h5_rgb/test_0.h5',
                        help='输入 HDF5 文件路径。')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='要处理的 HDF5 文件中的样本索引。')
    parser.add_argument('--checkpoint', type=str, default="checkpoints_seg_pyg_ptconv_rgb/best_model.pth",
                        help='预训练模型检查点 (.pth) 文件路径。')
    parser.add_argument('--model_file', type=str, default="stp/DIN_912-M8x30.stl",
                        help='Path to the target 3D model file (e.g., screw_model.ply/stl/obj) for ICP.')  # 新增
    parser.add_argument('--output_dir', type=str, default='./segmentation_results_h5_no_try',
                        help='保存输出文件的目录 (如果使用 --save_results)。')

    # 数据处理
    parser.add_argument('--num_points', type=int, default=40960,
                        help='期望的点数 (仅供参考, 代码会使用 HDF5 中的实际点数)。')
    parser.add_argument('--no_rgb', action='store_true', help='不加载或使用 RGB 数据。')
    parser.add_argument('--normalize', action='store_true', default=False, help='对加载的 XYZ 应用归一化。')

    # 模型参数
    parser.add_argument('--num_classes', type=int, default=2,
                        help='模型训练时的语义类别数量。')
    parser.add_argument('--k_neighbors', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--pt_hidden_dim', type=int, default=128)
    parser.add_argument('--pt_heads', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)

    # 聚类参数
    parser.add_argument('--do_clustering', action='store_true', default=True, help='启用 DBSCAN 聚类 (默认启用)。') # 改为默认启用
    parser.add_argument('--screw_label_id', type=int, default=0,
                        help='要进行 DBSCAN 聚类的目标语义标签 ID。')
    parser.add_argument('--dbscan_eps', type=float, default=20,
                        help='DBSCAN 的 eps 参数 (邻域半径)。需要根据点云密度和归一化后的尺度调整。')
    parser.add_argument('--dbscan_min_samples', type=int, default=10,
                        help='DBSCAN 的 min_samples 参数 (形成核心点的最小邻居数)。需要根据预期实例大小调整。')


    # ICP 参数
    parser.add_argument('--icp_threshold', type=float, default=5, help='ICP correspondence distance threshold (meters, default: 0.01 for 1cm). TUNABLE.') # 新增
    parser.add_argument('--icp_max_iter', type=int, default=10, help='ICP maximum iterations (default: 100).') # 新增

    # 控制参数
    parser.add_argument('--no_cuda', action='store_true', help='禁用 CUDA。')
    parser.add_argument('--save_results', action='store_true', help='保存 NPY/TXT 结果和最终姿态 (.npz)。')
    parser.add_argument('--no_visualize', action='store_true', help='禁用最终的 Open3D 场景可视化。')
    # 移除 marker_radius 和 visualize_pose

    args = parser.parse_args()
    if not SKLEARN_AVAILABLE: print("FATAL: scikit-learn required."); sys.exit(1)
    if not OPEN3D_AVAILABLE: print("FATAL: Open3D required."); sys.exit(1)
    model_input_dim_expected = 3 if args.no_rgb else 6; print(f"模型输入维度配置为: {model_input_dim_expected}D")
    main(args)
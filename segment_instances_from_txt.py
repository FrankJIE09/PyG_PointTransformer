# segment_instances_from_txt.py
# 从 TXT 文件加载点云，执行语义分割+聚类，可视化标记出的实例。

import torch
import numpy as np
import argparse
import time
import os
import random
import datetime
import sys

# --- 导入本地模块 ---
try:
    from model import PyG_PointTransformerSegModel # 语义分割模型类
except ImportError as e:
    print(f"FATAL Error importing model: {e}")
    print("Ensure model.py (with PyG_PointTransformerSegModel) is accessible.")
    sys.exit(1)

# --- 导入 Open3D 和 Scikit-learn ---
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("FATAL Error: Open3D not found. Visualization requires Open3D.")
    sys.exit(1)
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    print("FATAL Error: scikit-learn not found. Clustering requires scikit-learn.")
    sys.exit(1)


# --- 预处理函数 (适配文件加载) ---
def preprocess_file_point_cloud(points_np, num_target_points, coord_indices, rgb_indices=None, normalize=True):
    """
    从 NumPy 数组预处理点云，采样/填充，归一化，返回模型输入和处理后的坐标。

    Args:
        points_np (np.ndarray): 从文件加载的原始点云数据 (N_orig, C)。
        num_target_points (int): 模型期望的点数。
        coord_indices (list): XYZ 列的索引。
        rgb_indices (list | None): RGB 列的索引，如果为 None 则不使用颜色。
        normalize (bool): 是否进行归一化。

    Returns:
        torch.Tensor | None: 模型输入特征 (1, num_target_points, 6 or 3)。
        np.ndarray | None: 处理后的 XYZ 坐标 (num_target_points, 3)。
    """
    if points_np is None or points_np.ndim != 2 or points_np.shape[0] == 0:
        print("Error: Invalid input point cloud data.")
        return None, None

    # --- 提取坐标和颜色 ---
    try:
        points_xyz = points_np[:, coord_indices].astype(np.float32)
        points_rgb = None
        if rgb_indices:
            if max(rgb_indices) < points_np.shape[1]:
                points_rgb = points_np[:, rgb_indices].astype(np.uint8) # 假设原始 RGB 是 0-255
            else:
                print("Warning: RGB indices out of bounds, ignoring RGB.")
    except IndexError:
        print("Error: Coordinate or RGB column indices out of bounds.")
        return None, None
    except Exception as e:
        print(f"Error extracting columns: {e}")
        return None, None

    current_num_points = points_xyz.shape[0]

    # --- 采样/填充 ---
    if current_num_points == num_target_points:
        choice_idx = np.arange(current_num_points)
    elif current_num_points > num_target_points:
        choice_idx = np.random.choice(current_num_points, num_target_points, replace=False)
    else: # current_num_points < num_target_points
        choice_idx = np.random.choice(current_num_points, num_target_points, replace=True)

    processed_points_xyz = points_xyz[choice_idx, :]
    processed_rgb = points_rgb[choice_idx, :] if points_rgb is not None else None

    # --- 归一化 XYZ ---
    if normalize:
        centroid = np.mean(processed_points_xyz, axis=0)
        processed_points_xyz = processed_points_xyz - centroid
        max_dist = np.max(np.sqrt(np.sum(processed_points_xyz ** 2, axis=1)))
        if max_dist > 1e-6: processed_points_xyz = processed_points_xyz / max_dist

    # --- 准备模型输入特征 ---
    if processed_rgb is not None:
        rgb_normalized = processed_rgb.astype(np.float32) / 255.0
        features_np = np.concatenate((processed_points_xyz, rgb_normalized), axis=1) # (N, 6)
    else:
        features_np = processed_points_xyz # (N, 3) - 模型需要能处理 3D 输入

    # --- 转换为 Tensor ---
    features_tensor = torch.from_numpy(features_np).float().unsqueeze(0) # (1, N, 6 or 3)

    # 返回模型输入和处理后的 XYZ 坐标 (float32)
    return features_tensor, processed_points_xyz.astype(np.float32)


# --- 可视化函数 ---
def visualize_instances_with_markers(points_np, instance_labels, centroids, window_name="Instance Segmentation"):
    """
    使用 Open3D 可视化实例分割结果，并用球体标记实例中心。
    """
    if not OPEN3D_AVAILABLE: print("Open3D not available."); return
    if points_np is None or instance_labels is None or points_np.size == 0: return
    if points_np.shape[0] != instance_labels.shape[0]: return

    geometries = [] # 用于存储要显示的几何体

    # --- 1. 创建实例点云并着色 ---
    unique_instance_ids = np.unique(instance_labels) # 包括 -1 (噪声/背景)
    num_instances = len(unique_instance_ids[unique_instance_ids != -1])
    print(f"  Visualizing {num_instances} found instances and noise points.")

    np.random.seed(42) # 固定颜色映射
    # +1 for background/noise if -1 exists, +1 for range
    instance_colors = np.random.rand(int(np.max(instance_labels)) + 2, 3)
    instance_colors[0] = [0.5, 0.5, 0.5] # 噪声/背景用灰色 (映射 -1 到索引 0)

    # 创建一个包含所有点的点云，按实例着色
    pcd_instances = o3d.geometry.PointCloud()
    pcd_instances.points = o3d.utility.Vector3dVector(points_np)
    point_colors = instance_colors[instance_labels + 1] # instance_id + 1 作为颜色索引
    pcd_instances.colors = o3d.utility.Vector3dVector(point_colors)
    geometries.append(pcd_instances)

    # --- 2. 创建实例中心标记 (小球体) ---
    marker_radius = args.marker_radius # 从 args 获取半径
    marker_color = [1.0, 0.0, 0.0] # 标记用红色

    for inst_id, centroid in centroids.items():
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
        marker.paint_uniform_color(marker_color)
        marker.translate(centroid)
        geometries.append(marker)
        print(f"  Instance {inst_id} centroid marked at: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")


    # --- 3. 显示 ---
    print("Displaying instance segmentation result with centroid markers...")
    print("(Close the Open3D window to exit)")
    o3d.visualization.draw_geometries(geometries, window_name=window_name, width=1024, height=768)
    print("Visualization window closed.")


# --- 主函数 ---
def main(args):
    print(f"Starting instance segmentation at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    # --- 依赖检查 ---
    if not SKLEARN_AVAILABLE: print("FATAL Error: scikit-learn not found."); sys.exit(1)
    if not OPEN3D_AVAILABLE: print("FATAL Error: Open3D not found."); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"); print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True) # 创建输出目录

    # --- 加载模型 ---
    print(f"Loading SEMANTIC model checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint): raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    # 确定模型输入维度
    model_input_channels = 6 if not args.no_rgb else 3
    # --- !!! 注意: 需要修改 model.py 以支持 3D/6D 输入 !!! ---
    # --- 或者在这里根据 model_input_channels 修改 args ---
    # --- 假设 model.py 现在能处理 3D 或 6D ---
    print(f"Initializing model for {model_input_channels}D input features...")
    temp_args_for_model = argparse.Namespace(**vars(args)) # 复制 args
    # 临时修改或确认 model.py 能处理
    # 如果 model.py 必须是 6D，而输入是 3D，需要在 preprocess 中处理

    model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=temp_args_for_model).to(device)
    print("Model structure initialized.")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    else: model.load_state_dict(checkpoint)
    model.eval(); print("Model loaded successfully.")

    # --- 加载输入 TXT 文件 ---
    print(f"Loading input point cloud from: {args.input_txt}")
    if not os.path.exists(args.input_txt): raise FileNotFoundError(f"Input TXT file not found: {args.input_txt}")
    try:
        # 假设分隔符为空格或逗号
        raw_data_np = np.loadtxt(args.input_txt, dtype=np.float32)
        print(f"Loaded raw data with shape: {raw_data_np.shape}")
    except Exception as e:
        print(f"Error loading TXT file: {e}"); return

    # --- 预处理点云 ---
    print(f"Preprocessing point cloud to {args.num_points} points...")
    rgb_indices = [int(i.strip()) for i in args.rgb_cols.split(',')] if not args.no_rgb else None
    coord_indices = [int(i.strip()) for i in args.coord_cols.split(',')]
    features_tensor, points_processed_np = preprocess_file_point_cloud(
        raw_data_np, args.num_points, coord_indices, rgb_indices, args.normalize
    )
    if features_tensor is None or points_processed_np is None:
        print("Error during preprocessing."); return
    print(f"Preprocessing complete. Model input shape: {features_tensor.shape}, Processed points shape: {points_processed_np.shape}")

    # --- 语义分割推理 ---
    print("Performing semantic segmentation inference...")
    features_tensor = features_tensor.to(device)
    with torch.no_grad():
        logits = model(features_tensor) # 模型输入是 features_tensor
    pred_semantic_labels_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy() # (N,)
    print("Semantic prediction complete.")

    # --- DBSCAN 聚类 ---
    print("Performing DBSCAN clustering...")
    screw_label_id = args.screw_label_id
    screw_mask = (pred_semantic_labels_np == screw_label_id)
    screw_points_xyz = points_processed_np[screw_mask] # 使用处理后的 XYZ

    instance_labels_full = np.full_like(pred_semantic_labels_np, -1, dtype=np.int64)
    centroids = {} # 存储每个实例的中心 {inst_id: centroid_coord}
    num_instances_found = 0

    if screw_points_xyz.shape[0] >= args.dbscan_min_samples:
        db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
        instance_labels_screw_points = db.fit_predict(screw_points_xyz)
        instance_labels_full[screw_mask] = instance_labels_screw_points

        unique_instances, counts = np.unique(instance_labels_screw_points[instance_labels_screw_points != -1], return_counts=True)
        num_instances_found = len(unique_instances)
        num_noise_points = np.sum(instance_labels_screw_points == -1)
        print(f"Clustering found {num_instances_found} instances and {num_noise_points} noise points.")

        # 计算每个实例的中心点
        for i, inst_id in enumerate(unique_instances):
            instance_mask_local = (instance_labels_screw_points == inst_id)
            instance_points = screw_points_xyz[instance_mask_local]
            centroid = np.mean(instance_points, axis=0)
            centroids[inst_id] = centroid
            print(f"  Instance {inst_id}: {counts[i]} points, Centroid=[{centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f}]")

    else:
        print(f"Not enough points ({screw_points_xyz.shape[0]}) predicted as screw (label {screw_label_id}) for DBSCAN.")

    # --- 保存结果 (可选) ---
    if args.save_results:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        print(f"\nSaving results with timestamp {timestamp} to {args.output_dir}...")
        files_saved_msg = ["npy", "txt"]

        # 保存 NPY
        np.save(os.path.join(args.output_dir, f"points_{timestamp}.npy"), points_processed_np)
        np.save(os.path.join(args.output_dir, f"semantic_labels_{timestamp}.npy"), pred_semantic_labels_np)
        np.save(os.path.join(args.output_dir, f"instance_labels_{timestamp}.npy"), instance_labels_full)

        # 保存 TXT (XYZ, SemLabel, InstLabel)
        data_to_save = np.hstack((points_processed_np, pred_semantic_labels_np.reshape(-1, 1), instance_labels_full.reshape(-1, 1)))
        txt_filename = os.path.join(args.output_dir, f"segmented_full_{timestamp}.txt")
        np.savetxt(txt_filename, data_to_save, fmt='%.6f,%.6f,%.6f,%d,%d', delimiter=',')

        # 按实例分别保存 PLY
        num_ply_saved = 0
        if num_instances_found > 0:
            inst_colors = np.random.rand(num_instances_found + 1, 3) # 为实例 0, 1, ... 生成颜色
            inst_id_map = {inst_id: i for i, inst_id in enumerate(np.unique(instance_labels_full[instance_labels_full != -1]))} # 映射实例ID到颜色索引

            for inst_id in centroids.keys(): # 只保存找到的实例（非噪声）
                 mask = (instance_labels_full == inst_id)
                 points_for_instance = points_processed_np[mask]
                 pcd_inst = o3d.geometry.PointCloud()
                 pcd_inst.points = o3d.utility.Vector3dVector(points_for_instance)
                 label_color = inst_colors[inst_id_map[inst_id]] # 获取该实例的颜色
                 pcd_inst.paint_uniform_color(label_color)
                 ply_filename = os.path.join(args.output_dir, f"instance_{timestamp}_id_{inst_id}.ply")
                 try:
                     o3d.io.write_point_cloud(ply_filename, pcd_inst, write_ascii=False)
                     num_ply_saved += 1
                 except Exception as e_ply: print(f"\nError saving PLY for instance {inst_id}: {e_ply}")

        if num_ply_saved > 0: files_saved_msg.append(f"{num_ply_saved}_instance_ply")
        print(f"Saved output ({', '.join(files_saved_msg)})")

    # --- 可视化结果 ---
    if not args.no_visualize:
        visualize_instances_with_markers(points_processed_np, instance_labels_full, centroids)

    print("\nInstance segmentation from file finished.")


# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform instance segmentation (Semantic+Clustering) on a TXT point cloud file.')

    # --- 输入/输出 ---
    parser.add_argument('--input_txt', type=str, default="pcl.txt", help='Path to the input .txt point cloud file.')
    parser.add_argument('--checkpoint', type=str, default="checkpoints_seg_pyg_ptconv_rgb/best_model.pth",
                        help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, default='./instance_results', help='Directory to save output files (if --save_results is used).')

    # --- 数据处理 ---
    parser.add_argument('--num_points', type=int, default=20480, help='Target number of points for model input (sampling/padding).')
    parser.add_argument('--coord_cols', type=str, default='0,1,2', help='Indices for X, Y, Z columns in TXT (0-based).')
    parser.add_argument('--rgb_cols', type=str, default='3,4,5', help='Indices for R, G, B columns in TXT (0-based).')
    parser.add_argument('--no_rgb', action='store_true', help='Input TXT file does not contain RGB columns.')
    parser.add_argument('--normalize', action='store_true', default=True, help='Apply normalization (center + unit sphere) during preprocessing.') # 默认开启归一化

    # --- 模型参数 (必须与训练时一致) ---
    parser.add_argument('--num_classes', type=int, default="2", help='Number of SEMANTIC classes model was trained for.')
    parser.add_argument('--k_neighbors', type=int, default=16, help='(Model Arch) k for k-NN graph.')
    parser.add_argument('--embed_dim', type=int, default=64, help='(Model Arch) Initial embedding dimension.')
    parser.add_argument('--pt_hidden_dim', type=int, default=128, help='(Model Arch) Hidden dimension for PointTransformerConv.')
    parser.add_argument('--pt_heads', type=int, default=4, help='(Model Arch) Number of attention heads.')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='(Model Arch) Number of PointTransformerConv layers.')
    parser.add_argument('--dropout', type=float, default=0.3, help='(Model Arch) Dropout rate.')

    # --- 聚类参数 ---
    # --do_clustering is implicitly True for this script
    parser.add_argument('--screw_label_id', type=int, default=1, help='The semantic label ID for screws (default: 1).')
    parser.add_argument('--dbscan_eps', type=float, default=2, help='DBSCAN eps parameter (meters, default: 0.02). TUNABLE.')
    parser.add_argument('--dbscan_min_samples', type=int, default=100, help='DBSCAN min_samples parameter (default: 10). TUNABLE.')

    # --- 控制参数 ---
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA.')
    parser.add_argument('--save_results', action='store_true', help='Save output arrays (NPY, TXT) and per-instance PLY files.')
    parser.add_argument('--no_visualize', action='store_true', help='Disable final Open3D visualization.')
    parser.add_argument('--marker_radius', type=float, default=0.01, help='Radius of the spheres marking instance centroids in visualization (meters, default: 0.01).')


    args = parser.parse_args()

    # --- 依赖检查 ---
    if not SKLEARN_AVAILABLE: print("FATAL Error: scikit-learn is required for clustering."); sys.exit(1)
    if not args.no_visualize and not OPEN3D_AVAILABLE: print("FATAL Error: Open3D is required for visualization."); sys.exit(1)
    if args.save_results and not OPEN3D_AVAILABLE: print("Warning: Open3D not available, cannot save per-instance PLY files."); # 继续保存 NPY/TXT

    # 解析列索引
    try:
        args.coord_indices = [int(i.strip()) for i in args.coord_cols.split(',')]
        assert len(args.coord_indices) == 3
        if not args.no_rgb:
            args.rgb_indices = [int(i.strip()) for i in args.rgb_cols.split(',')]
            assert len(args.rgb_indices) == 3
        else:
            args.rgb_indices = None
    except Exception as e:
        print(f"Error parsing column indices: {e}"); sys.exit(1)

    # --- !!! 模型输入维度适配 !!! ---
    # 检查模型是否需要 6D 输入，但数据只有 3D
    if not args.no_rgb:
        model_input_dim = 6
        print("Assuming model expects 6D input (XYZRGB).")
    else:
        model_input_dim = 3
        print("Assuming model expects 3D input (XYZ).")
        # !!! 重要: 如果模型实际上需要 6D (例如 MLP 输入层是 nn.Linear(6, ...)),
        # !!! 但这里只有 3D，那么 preprocess_file_point_cloud 需要能够处理这种情况
        # !!! (例如，通过添加默认颜色来凑成 6D)，或者 model.py 需要能处理 3D 输入。
        # !!! 我们当前的 preprocess 函数在输入只有 XYZ 时会添加默认颜色，所以应该还好。
        # !!! 但最好的方式是 model.py 本身就支持可变输入维度，或者根据参数调整。

    main(args)
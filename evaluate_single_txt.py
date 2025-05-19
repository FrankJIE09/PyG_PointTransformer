# evaluate_single_txt.py
# 加载单个 TXT 文件（含标签），进行语义分割预测，计算指标并可视化。

import torch
import numpy as np
import argparse
import time
import os
import datetime
import sys

# --- 导入本地模块 ---
try:
    from model import PyG_PointTransformerSegModel # 语义分割模型类
except ImportError as e: print(f"FATAL Error importing model: {e}"); sys.exit(1)

# --- 导入 Open3D 和 Scikit-learn (如果需要) ---
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError: print("Warning: Open3D not found. Visualization disabled."); OPEN3D_AVAILABLE = False
# DBSCAN 在这个脚本里不需要，但如果以后要加聚类可以取消注释
try: from sklearn.cluster import DBSCAN
except ImportError: print("Warning: scikit-learn not found."); pass


# --- 预处理函数 (适配单个 TXT 文件) ---
def preprocess_single_txt_data(raw_data_np, num_target_points, coord_indices, label_col, rgb_indices=None, normalize=True):
    """
    从加载的 TXT NumPy 数组预处理点云，采样/填充，归一化。
    返回: 模型输入特征 Tensor (1,N,C), 处理后的 XYZ NumPy (N,3), 处理后的真实标签 NumPy (N,)
    """
    if raw_data_np is None or raw_data_np.ndim != 2 or raw_data_np.shape[0] == 0:
        print("Error: Invalid raw data numpy array.")
        return None, None, None

    required_cols = max(max(coord_indices), (max(rgb_indices) if rgb_indices else -1), label_col)
    if raw_data_np.shape[1] <= required_cols:
        print(f"Error: Insufficient columns in TXT. Need at least {required_cols+1}, got {raw_data_np.shape[1]}.")
        return None, None, None

    try:
        points_xyz = raw_data_np[:, coord_indices].astype(np.float32)
        gt_labels = raw_data_np[:, label_col].astype(np.int64) # 真实标签
        points_rgb = None
        features_to_combine = [points_xyz]
        if rgb_indices:
            points_rgb_float = raw_data_np[:, rgb_indices].astype(np.float32)
            points_rgb_normalized = np.clip(points_rgb_float / 255.0, 0.0, 1.0)
            features_to_combine.append(points_rgb_normalized)
        elif not args.no_rgb: # 模型需要 6D 但输入只有 XYZ+Label
             print("Warning: Model expects RGB but --no_rgb not set or rgb_cols invalid. Adding default gray.")
             default_colors = np.full((points_xyz.shape[0], 3), 0.5, dtype=np.float32)
             features_to_combine.append(default_colors) # 添加默认颜色

    except IndexError: print("Error: Column indices out of bounds."); return None, None, None
    except ValueError as e: print(f"Error converting columns: {e}"); return None, None, None

    if points_xyz.shape[0] != gt_labels.shape[0]: print("Error: Points and labels count mismatch."); return None, None, None

    current_num_points = points_xyz.shape[0]

    # --- 采样/填充 (同时应用于坐标、颜色(如果存在)、标签) ---
    if current_num_points == num_target_points:
        choice_idx = np.arange(current_num_points)
    elif current_num_points > num_target_points:
        choice_idx = np.random.choice(current_num_points, num_target_points, replace=False)
    else:
        choice_idx = np.random.choice(current_num_points, num_target_points, replace=True)

    processed_points_xyz = points_xyz[choice_idx, :]
    processed_labels = gt_labels[choice_idx]
    processed_features_list = [processed_points_xyz]
    if len(features_to_combine) > 1:
        processed_rgb = features_to_combine[1][choice_idx, :]
        processed_features_list.append(processed_rgb)

    # --- 归一化 XYZ ---
    if normalize:
        centroid = np.mean(processed_points_xyz, axis=0)
        processed_points_xyz = processed_points_xyz - centroid
        max_dist = np.max(np.sqrt(np.sum(processed_points_xyz ** 2, axis=1)))
        if max_dist > 1e-6: processed_points_xyz = processed_points_xyz / max_dist
        processed_features_list[0] = processed_points_xyz # 更新列表中的坐标

    features_np = np.concatenate(processed_features_list, axis=1)
    features_tensor = torch.from_numpy(features_np).float().unsqueeze(0)

    return features_tensor, processed_points_xyz.astype(np.float32), processed_labels


# --- 可视化函数 (与 _evaluate.py 相同) ---
def visualize_combined_prediction(points_np, pred_labels_np, num_classes, file_name):
    if not OPEN3D_AVAILABLE: print("Open3D not available."); return
    if points_np is None or pred_labels_np is None or points_np.size == 0: return
    if points_np.shape[0] != pred_labels_np.shape[0]: return
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])
    np.random.seed(42); colors = np.random.rand(num_classes, 3)
    try: clamped_labels = np.clip(pred_labels_np, 0, num_classes - 1); point_colors = colors[clamped_labels]
    except IndexError as e: print(f"Error coloring: {e}"); point_colors = np.random.rand(points_np.shape[0], 3)
    pcd.colors = o3d.utility.Vector3dVector(point_colors); window_title = f"Prediction - {os.path.basename(file_name)}"
    print(f"\nDisplaying combined semantic prediction for {os.path.basename(file_name)}...")
    o3d.visualization.draw_geometries([pcd], window_name=window_title, width=800, height=600)
    print("Visualization window closed.")

# --- 工具函数 - 计算指标 (与 _evaluate.py 相同) ---
def calculate_metrics_overall(pred_labels_all, target_labels_all, num_classes):
    total_points = target_labels_all.size; correct_points = np.sum(pred_labels_all == target_labels_all)
    overall_accuracy = (correct_points / total_points) * 100.0 if total_points > 0 else 0.0
    intersection = np.zeros(num_classes); union = np.zeros(num_classes)
    for cl in range(num_classes):
        pred_inds = (pred_labels_all == cl); target_inds = (target_labels_all == cl)
        intersection[cl] = np.logical_and(pred_inds, target_inds).sum(); union[cl] = np.logical_or(pred_inds, target_inds).sum()
    iou_per_class = np.full(num_classes, np.nan); has_union = (union > 0); iou_per_class[has_union] = intersection[has_union] / union[has_union]
    mIoU = np.nanmean(iou_per_class) * 100.0 if np.any(~np.isnan(iou_per_class)) else 0.0
    return overall_accuracy, mIoU, iou_per_class * 100.0


# --- 主函数 ---
def main(args):
    print(f"Starting evaluation for single file at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    if not args.no_visualize and not OPEN3D_AVAILABLE: print("Error: Visualization requested but Open3D not available."); sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"); print(f"Using device: {device}")

    # --- 加载模型 ---
    print(f"Loading model checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint): raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model_input_channels = 3 if args.no_rgb else 6
    print(f"Initializing model for {model_input_channels}D input features...")
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

    # --- 加载 TXT 数据 ---
    print(f"Loading input point cloud from: {args.input_txt}")
    if not os.path.exists(args.input_txt): raise FileNotFoundError(f"Input TXT file not found: {args.input_txt}")
    try: raw_data_np = np.loadtxt(args.input_txt, dtype=np.float32); print(f"Loaded raw data with shape: {raw_data_np.shape}")
    except Exception as e: print(f"Error loading TXT file: {e}"); return

    # --- 预处理 ---
    print(f"Preprocessing point cloud to {args.num_points} points...")
    rgb_indices = [int(i.strip()) for i in args.rgb_cols.split(',')] if not args.no_rgb else None
    coord_indices = [int(i.strip()) for i in args.coord_cols.split(',')]
    label_col_idx = args.label_col
    features_tensor, points_processed_np, gt_labels_np = preprocess_single_txt_data(
        raw_data_np, args.num_points, coord_indices, label_col_idx, rgb_indices, args.normalize
    )
    if features_tensor is None: print("Error during preprocessing."); return
    print(f"Preprocessing complete. Model input: {features_tensor.shape}, Processed points: {points_processed_np.shape}, GT labels: {gt_labels_np.shape}")
    if features_tensor.shape[-1] != model_input_channels:
         print(f"FATAL ERROR: Model expects {model_input_channels}D features, preprocessor generated {features_tensor.shape[-1]}D."); sys.exit(1)

    # --- 模型推理 ---
    print("Performing semantic segmentation inference...")
    features_tensor = features_tensor.to(device)
    with torch.no_grad(): logits = model(features_tensor) # (1, N, num_classes)
    pred_labels_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy() # (N,)
    print("Inference complete.")

    # --- 计算指标 ---
    print("Calculating metrics for this file...")
    # 注意：这里的 pred_labels_np 和 gt_labels_np 已经是处理后的 N 个点的标签
    overall_accuracy, mIoU, iou_per_class = calculate_metrics_overall(
        pred_labels_np, gt_labels_np, args.num_classes
    )

    # --- 打印结果 ---
    print("\n--- Evaluation Results (Single File) ---")
    print(f"Input File: {os.path.basename(args.input_txt)}")
    print(f"Overall Point Accuracy: {overall_accuracy:.2f}%")
    print(f"Mean IoU (mIoU): {mIoU:.2f}%")
    print("\nIoU per class:")
    for i in range(args.num_classes):
        iou_val = iou_per_class[i]; print(f"  Class {i:2d}: {iou_val:.2f}%" if not np.isnan(iou_val) else f"  Class {i:2d}: NaN")
    print("-" * 25)

    # --- 可视化预测结果 ---
    if not args.no_visualize:
        visualize_combined_prediction(points_processed_np, pred_labels_np, args.num_classes, args.input_txt)

    print("\nEvaluation for single file finished.")


# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained segmentation model on a single TXT point cloud file.')

    # --- 输入/输出 ---
    parser.add_argument('--input_txt', type=str, default='data/12345678/point_cloud_00003.txt',  # 设置默认值
                        help='Path to the input .txt point cloud file (must include GT labels) (default: pcl.txt)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_seg_pyg_ptconv_rgb/best_model.pth',  # 设置默认值
                        help='Path to the trained model checkpoint (.pth) (default: checkpoints_seg_pyg_ptconv_rgb/best_model.pth)')

    # --- 数据处理 ---
    parser.add_argument('--num_points', type=int, default=2048, help='Target number of points for model input (sampling/padding).')
    parser.add_argument('--coord_cols', type=str, default='0,1,2', help='Indices for X, Y, Z columns in TXT.')
    # 标签列现在必需
    parser.add_argument('--label_col', type=int, default=6, help='Index of the ground truth segmentation label column in TXT (0-based).')
    parser.add_argument('--rgb_cols', type=str, default='3,4,5', help='Indices for R, G, B columns in TXT (if available).')
    parser.add_argument('--no_rgb', action='store_true', help='Input TXT file does not contain RGB columns (model will get XYZ input).')
    parser.add_argument('--normalize', action='store_true', default=False, help='Apply normalization (center + unit sphere) during preprocessing.')

    # --- 模型参数 ---
    parser.add_argument('--num_classes', type=int, default=2, help='Number of SEMANTIC classes model was trained for.')
    parser.add_argument('--k_neighbors', type=int, default=16, help='(Model Arch) k for k-NN graph.')
    parser.add_argument('--embed_dim', type=int, default=64, help='(Model Arch) Initial embedding dimension.')
    parser.add_argument('--pt_hidden_dim', type=int, default=128, help='(Model Arch) Hidden dimension for PointTransformerConv.')
    parser.add_argument('--pt_heads', type=int, default=4, help='(Model Arch) Number of attention heads.')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='(Model Arch) Number of PointTransformerConv layers.')
    parser.add_argument('--dropout', type=float, default=0.3, help='(Model Arch) Dropout rate.')

    # --- 控制参数 ---
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA.')
    parser.add_argument('--no_visualize', action='store_true', help='Disable final Open3D visualization.')

    args = parser.parse_args()

    # --- 依赖检查 ---
    if not args.no_visualize and not OPEN3D_AVAILABLE: print("Error: Visualization requested but Open3D not available."); sys.exit(1)

    # --- 解析列索引 ---
    try:
        args.coord_indices = [int(i.strip()) for i in args.coord_cols.split(',')]; assert len(args.coord_indices) == 3
        if not args.no_rgb: args.rgb_indices = [int(i.strip()) for i in args.rgb_cols.split(',')]; assert len(args.rgb_indices) == 3
        else: args.rgb_indices = None
        if args.label_col < 0: raise ValueError("Label column index must be non-negative")
    except Exception as e: print(f"Error parsing column indices: {e}"); sys.exit(1)

    # --- 推断模型输入维度并检查兼容性 ---
    model_input_dim_expected = 3 if args.no_rgb else 6
    print(f"Script configured for {model_input_dim_expected}D model input features.")
    # 可以在这里添加对 model.py 中第一层 Linear 权重的检查来确认模型实际维度，但比较复杂
    # 暂时依赖用户确保模型与 --no_rgb 参数匹配

    main(args)
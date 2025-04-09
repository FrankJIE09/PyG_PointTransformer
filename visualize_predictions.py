# visualize_predictions.py
# 加载 HDF5 样本和训练好的模型，进行预测，并按预测类别逐个显示点云。

import torch
import numpy as np
import argparse
import os
import sys
import h5py
import time # 可选，用于可能的延迟
import datetime
# --- 导入本地模块 ---
try:
    # 确保 model.py 在同一目录或 Python 路径中
    from model import PyG_PointTransformerSegModel # 使用的模型类
except ImportError as e:
    print(f"FATAL Error importing model class: {e}")
    sys.exit(1)

# --- 导入 Open3D ---
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("FATAL Error: Open3D not found. This script requires Open3D.")
    sys.exit(1)


def main(args):
    print(f"Starting prediction visualization at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    # --- 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- 1. 加载 HDF5 数据样本 ---
    print(f"Loading data from: {args.h5_file}, sample index: {args.sample_index}")
    if not os.path.exists(args.h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {args.h5_file}")

    try:
        with h5py.File(args.h5_file, 'r') as f:
            if 'data' not in f or 'seg' not in f:
                raise KeyError("HDF5 file must contain 'data' and 'seg' keys.")
            num_samples_in_file = f['data'].shape[0]
            if not (0 <= args.sample_index < num_samples_in_file):
                raise IndexError(f"Sample index {args.sample_index} out of bounds for file with {num_samples_in_file} samples.")

            # 加载指定样本的坐标和真实标签
            # HDF5 文件中的数据已经是处理过的 (num_points, 3)
            points_np = f['data'][args.sample_index].astype(np.float64) # Open3D 偏好 float64
            gt_labels_np = f['seg'][args.sample_index].astype(np.int64) # 真实标签
            print(f"Loaded sample {args.sample_index} with {points_np.shape[0]} points.")
            if points_np.shape[0] != args.num_points:
                 print(f"Warning: Number of points in HDF5 ({points_np.shape[0]}) does not match --num_points ({args.num_points}). Using data as is.")
                 # 更新 num_points 以匹配实际数据，供后续使用
                 args.num_points = points_np.shape[0]

    except Exception as e:
        print(f"Error loading data from HDF5: {e}")
        return

    # --- 2. 加载训练好的模型 ---
    print(f"Loading model checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # 实例化模型结构 (需要所有训练时的超参数)
    model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
    print("Model structure initialized.")

    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    else: model.load_state_dict(checkpoint)
    model.eval() # 设置为评估模式
    print("Model loaded successfully.")

    # --- 3. 预处理输入 ---
    # 将 NumPy 点云转换为模型所需的 Tensor 格式 (1, N, 3)
    # 注意：这里假设 HDF5 中的点云已经是采样/填充到 num_points 并且无需归一化
    points_tensor = torch.from_numpy(points_np).float().unsqueeze(0).to(device)

    # --- 4. 模型推理 ---
    print("Performing model inference...")
    with torch.no_grad():
        logits = model(points_tensor) # (1, num_points, num_classes)
    predictions = torch.argmax(logits, dim=2) # (1, num_points)
    pred_labels_np = predictions.squeeze(0).cpu().numpy() # (num_points,)
    print("Inference complete.")

    # --- 5. 按预测类别逐个显示 ---
    unique_predicted_labels = np.unique(pred_labels_np)
    print(f"\nFound {len(unique_predicted_labels)} unique predicted labels in this sample: {sorted(unique_predicted_labels)}")
    print("Displaying points for each predicted label sequentially...")
    print("(Close each Open3D window to proceed to the next label)")

    # --- 预生成颜色映射表 ---
    np.random.seed(42) # 固定种子以获得一致的颜色
    label_to_color = np.random.rand(args.num_classes, 3)

    for label_id in sorted(unique_predicted_labels): # 按标签顺序显示
        print(f"\nDisplaying points predicted as Label: {label_id}")

        # a. 筛选出属于当前预测类别的点
        mask = (pred_labels_np == label_id)
        points_for_label = points_np[mask]

        if points_for_label.shape[0] == 0:
            print("  (No points found for this label - should not happen if label is in unique_predicted_labels)")
            continue

        print(f"  Number of points for label {label_id}: {points_for_label.shape[0]}")

        # b. 创建新的 Open3D 点云对象
        pcd_label = o3d.geometry.PointCloud()
        pcd_label.points = o3d.utility.Vector3dVector(points_for_label)

        # c. 给这个类别的点云统一上色
        label_color = label_to_color[label_id % args.num_classes] # 使用预生成的颜色
        pcd_label.paint_uniform_color(label_color)

        # d. 显示这个只包含单一预测类别的点云 (阻塞)
        window_title = f"Prediction - Label {label_id} ({points_for_label.shape[0]} points)"
        o3d.visualization.draw_geometries([pcd_label], window_name=window_title, width=800, height=600)
        print(f"  Closed window for Label {label_id}.")
        # time.sleep(0.5) # 可选：短暂暂停，避免窗口关闭太快

    print("\nFinished displaying all predicted labels for this sample.")

    # (可选) 显示原始带 Groud Truth 颜色的点云以供对比
    if args.show_ground_truth:
        print("\nDisplaying Ground Truth segmentation for comparison...")
        visualize_ground_truth(points_np, gt_labels_np, window_name=f"Ground Truth - Sample {args.sample_index}")


# --- (从 evaluate.py 复制过来的可视化函数) ---
def visualize_ground_truth(pcd_points_np, pcd_labels_np, window_name="Ground Truth Segmentation"):
    # ... (代码与 evaluate.py 中的 visualize_ground_truth 完全相同) ...
    if not OPEN3D_AVAILABLE: print("Open3D not available."); return
    if pcd_points_np is None or pcd_labels_np is None: return
    if pcd_points_np.shape[0] != pcd_labels_np.shape[0]: return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points_np)
    num_classes_in_sample = int(np.max(pcd_labels_np)) + 1
    np.random.seed(42)
    colors = np.random.rand(num_classes_in_sample, 3)
    try: point_colors = colors[pcd_labels_np]
    except IndexError: clamped_labels = np.clip(pcd_labels_np, 0, num_classes_in_sample - 1); point_colors = colors[clamped_labels]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    print("Displaying point cloud colored by ground truth labels...")
    o3d.visualization.draw_geometries([pcd], window_name=window_name, width=800, height=600)
    print("Ground truth visualization window closed.")


# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize model predictions on a single HDF5 sample, showing one predicted class at a time.')

    # --- 必要参数 ---
    parser.add_argument('--h5_file', type=str,default="./data/my_custom_dataset_h5/train_0.h5", help='Path to the input HDF5 file (e.g., test_0.h5)')
    parser.add_argument('--checkpoint', type=str, default="./checkpoints_seg_pyg_ptconv/best_model.pth", help='Path to the trained model checkpoint (.pth file)')

    # --- 数据和模型参数 (必须与训练时一致!) ---
    parser.add_argument('--sample_index', type=int, default=0, help='Index of the sample within the HDF5 file to visualize (default: 0)')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points the model expects')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of segmentation classes model was trained for')
    parser.add_argument('--k_neighbors', type=int, default=16, help='(Model Arch) k for k-NN graph')
    parser.add_argument('--embed_dim', type=int, default=64, help='(Model Arch) Initial embedding dimension')
    parser.add_argument('--pt_hidden_dim', type=int, default=128, help='(Model Arch) Hidden dimension for PointTransformerConv')
    parser.add_argument('--pt_heads', type=int, default=4, help='(Model Arch) Number of attention heads')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='(Model Arch) Number of PointTransformerConv layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='(Model Arch) Dropout rate')

    # --- 其他参数 ---
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA inference (use CPU)')
    parser.add_argument('--show_ground_truth', action='store_true', help='Also show the ground truth segmentation at the end for comparison.')


    args = parser.parse_args()

    if not OPEN3D_AVAILABLE:
         print("FATAL ERROR: Open3D is required for this script.")
    else:
         main(args)



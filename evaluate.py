# evaluate.py
# 版本: 适配加载 6D 特征 HDF5 数据进行评估，可选可视化

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
import datetime
import random
import glob # 需要 glob
import sys

# --- 导入本地模块 ---
try:
    # 确保 dataset.py 是能返回 6D features 的版本
    from dataset import ShapeNetPartSegDataset
    # 确保 model.py 是能处理 6D features 的版本
    from model import PyG_PointTransformerSegModel
except ImportError as e:
    print(f"FATAL Error importing local modules (dataset.py, model.py): {e}")
    sys.exit(1)

# --- 导入 Open3D ---
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: Open3D not found. Visualization options will be disabled.")
    OPEN3D_AVAILABLE = False


# --- 可视化函数 (修改为接收 6D 特征但只用 XYZ) ---
def visualize_sample_by_class(features_np, pred_labels_np, num_classes, sample_idx, label_to_color):
    """ 按预测类别逐个显示单个样本的点云 (只使用 XYZ)。"""
    if not OPEN3D_AVAILABLE: print("Open3D not available."); return

    # 从 6D 特征中提取 XYZ 坐标用于可视化
    points_np = features_np[:, :3] # (N, 3)

    unique_predicted_labels = np.unique(pred_labels_np)
    print(f"\n--- Visualizing Sample Index: {sample_idx} ---")
    print(f"Predicted labels present: {sorted(unique_predicted_labels)}")
    print("(Close each Open3D window to see the next predicted class)")

    for label_id in sorted(unique_predicted_labels):
        print(f"  Showing points predicted as Label: {label_id}")
        mask = (pred_labels_np == label_id)
        points_for_label = points_np[mask]
        if points_for_label.shape[0] == 0: continue

        pcd_label = o3d.geometry.PointCloud()
        pcd_label.points = o3d.utility.Vector3dVector(points_for_label)
        label_color = label_to_color[label_id % num_classes]
        pcd_label.paint_uniform_color(label_color)

        window_title = f"Sample {sample_idx} - Predicted Label {label_id} ({points_for_label.shape[0]} points)"
        o3d.visualization.draw_geometries([pcd_label], window_name=window_title, width=800, height=600)
        print(f"  Closed window for Label {label_id}.")

    print(f"--- Finished visualizing sample {sample_idx} ---")


# --- 工具函数 - 计算指标 (保持不变) ---
def calculate_metrics_overall(pred_labels_all, target_labels_all, num_classes):
    # ... (代码同上一个 evaluate.py 版本) ...
    total_points = target_labels_all.size; correct_points = np.sum(pred_labels_all == target_labels_all)
    overall_accuracy = (correct_points / total_points) * 100.0 if total_points > 0 else 0.0
    intersection = np.zeros(num_classes); union = np.zeros(num_classes)
    for cl in range(num_classes):
        pred_inds = (pred_labels_all == cl); target_inds = (target_labels_all == cl)
        intersection[cl] = np.logical_and(pred_inds, target_inds).sum(); union[cl] = np.logical_or(pred_inds, target_inds).sum()
    iou_per_class = np.full(num_classes, np.nan)
    has_union = (union > 0); iou_per_class[has_union] = intersection[has_union] / union[has_union]
    mIoU = np.nanmean(iou_per_class) * 100.0 if np.any(~np.isnan(iou_per_class)) else 0.0
    return overall_accuracy, mIoU, iou_per_class * 100.0


# --- 评估函数 (适配 6D 特征, 修改可视化调用) ---
def run_evaluation(model, dataloader, device, num_classes, indices_to_visualize, label_to_color, args):
    model.eval()
    all_pred_labels_list = []
    all_target_labels_list = []
    total_processed_points = 0

    print(f"\nEvaluating on '{args.partition}' partition...")
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating on {args.partition}")
        for batch_idx, (features, seg_labels) in pbar: # 解包 features
            features, seg_labels = features.to(device), seg_labels.to(device) # (B, N, 6), (B, N)
            batch_size = features.shape[0]

            logits = model(features) # 输入 6D features
            predictions = torch.argmax(logits, dim=2) # (B, N)

            current_preds_flat = predictions.cpu().numpy().flatten()
            current_labels_flat = seg_labels.cpu().numpy().flatten()
            all_pred_labels_list.append(current_preds_flat)
            all_target_labels_list.append(current_labels_flat)
            total_processed_points += len(current_labels_flat)

            # --- 检查可视化 ---
            if indices_to_visualize:
                for i in range(batch_size):
                    sample_idx_global = batch_idx * dataloader.batch_size + i
                    if sample_idx_global in indices_to_visualize:
                        print(f"\nVisualizing sample at global index: {sample_idx_global}")
                        # !!! 修改点: 提取 XYZ 用于可视化 !!!
                        sample_features_np = features[i].cpu().numpy() # (N, 6)
                        sample_points_np = sample_features_np[:, :3]   # (N, 3) - 只取 XYZ
                        sample_pred_labels_np = predictions[i].cpu().numpy() # (N,)
                        visualize_sample_by_class( # 调用修改后的函数
                            sample_points_np,      # 传入 XYZ
                            sample_pred_labels_np,
                            num_classes,
                            sample_idx_global,
                            label_to_color
                        )
                        # indices_to_visualize.remove(sample_idx_global) # 可选移除

    print(f"\nEvaluation loop finished. Total points processed: {total_processed_points}")
    if total_processed_points == 0: return 0.0, 0.0, np.full(num_classes, np.nan)

    all_pred_labels = np.concatenate(all_pred_labels_list)
    all_target_labels = np.concatenate(all_target_labels_list)
    print(f"Calculating final metrics...")
    overall_accuracy, mIoU, iou_per_class = calculate_metrics_overall(
        all_pred_labels, all_target_labels, num_classes
    )
    return overall_accuracy, mIoU, iou_per_class


# --- 主函数 ---
def main(args):
    print(f"Starting evaluation at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    if args.visualize_n_samples > 0 and not OPEN3D_AVAILABLE:
        print("Warning: Visualization requested but Open3D not available. Disabling visualization.")
        args.visualize_n_samples = 0

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"); print(f"Using device: {device}")

    print(f"Loading dataset from: {args.data_root}")
    # 使用修改后的 Dataset (返回 6D features)
    try: # 保留这里的 try-except
        eval_dataset = ShapeNetPartSegDataset(data_root=args.data_root, partition=args.partition, num_points=args.num_points, augment=False)
    except Exception as e: print(f"Error loading dataset: {e}"); return
    if len(eval_dataset) == 0: print(f"Error: No data loaded for partition '{args.partition}'."); return

    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    print(f"Loaded '{args.partition}' dataset with {len(eval_dataset)} samples.")

    print(f"Loading model checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint): raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    # 使用修改后的 Model (接受 6D features)
    try: # 保留这里的 try-except
        model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
        print("Model structure initialized (expects 6D input).")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    except Exception as e: print(f"Error loading model: {e}"); return

    # 准备可视化
    indices_to_visualize = set()
    if args.visualize_n_samples > 0:
        if args.visualize_n_samples > len(eval_dataset): args.visualize_n_samples = len(eval_dataset)
        random.seed(args.seed); indices_to_visualize = set(random.sample(range(len(eval_dataset)), args.visualize_n_samples))
        print(f"Will visualize predictions for {len(indices_to_visualize)} randomly selected samples: {sorted(list(indices_to_visualize))}")

    np.random.seed(42); label_to_color = np.random.rand(args.num_classes, 3)

    # 执行评估
    start_eval_time = time.time()
    overall_accuracy, mIoU, iou_per_class = run_evaluation( # 调用修改后的函数名
        model, eval_loader, device, args.num_classes, indices_to_visualize, label_to_color, args
    )
    end_eval_time = time.time()

    # 打印结果
    print("\n--- Evaluation Results ---")
    print(f"Partition Evaluated: {args.partition}")
    print(f"Overall Point Accuracy: {overall_accuracy:.2f}%")
    print(f"Mean IoU (mIoU): {mIoU:.2f}%")
    print("\nIoU per class:")
    for i in range(args.num_classes):
        iou_val = iou_per_class[i]; print(f"  Class {i:2d}: {iou_val:.2f}%" if not np.isnan(iou_val) else f"  Class {i:2d}: NaN")
    print("-" * 25)
    print(f"Evaluation completed in {end_eval_time - start_eval_time:.2f} seconds.")


# --- 命令行参数解析 (与 train.py 保持模型参数一致) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained PyG PointTransformer Segmentation model (XYZRGB input).')

    parser.add_argument('--checkpoint', type=str, default="checkpoints_seg_pyg_ptconv_rgb/best_model.pth", help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--data_root', type=str, default='./data/my_custom_dataset_h5_rgb', # 使用含 RGB 的 HDF5 目录
                        help='Path to the directory containing HDF5 files (with data, rgb, seg keys)')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points the model expects')
    parser.add_argument('--num_classes', type=int, default=50, help='Number of segmentation classes model was trained for')
    parser.add_argument('--k_neighbors', type=int, default=16, help='(Model Arch) k for k-NN graph')
    parser.add_argument('--embed_dim', type=int, default=64, help='(Model Arch) Initial embedding dimension')
    parser.add_argument('--pt_hidden_dim', type=int, default=128, help='(Model Arch) Hidden dimension for PointTransformerConv')
    parser.add_argument('--pt_heads', type=int, default=4, help='(Model Arch) Number of attention heads')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='(Model Arch) Number of PointTransformerConv layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='(Model Arch) Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--partition', type=str, default='test', choices=['train', 'val', 'test'], help="Which partition to evaluate (default: 'test')")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA evaluation (use CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for selecting visualization samples')
    parser.add_argument('--visualize_n_samples', type=int, default=1, metavar='N', help='Randomly select N samples to visualize class-by-class (default: 0)')

    args = parser.parse_args()

    if args.visualize_n_samples > 0 and not OPEN3D_AVAILABLE:
        print("Error: Visualization requested (--visualize_n_samples > 0) but Open3D is not available.")
        sys.exit(1)

    main(args)
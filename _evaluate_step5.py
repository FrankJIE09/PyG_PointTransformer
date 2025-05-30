# _evaluate_step5.py
# 版本: 适配加载 6D 特征 HDF5 数据进行评估，可选可视化
# 注意: 请根据您新的 _train_step.py 文件，仔细检查并调整 argparse 中的模型架构参数！

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
import glob
import sys
import yaml

# --- Helper function to load config from YAML (same as in other steps) ---
def load_config_from_yaml(config_path):
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            print(f"Successfully loaded configuration from {config_path}")
            return config_data
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML file {config_path}: {e}. Using script's default arguments.")
            return {}
        except Exception as e:
            print(f"Warning: Could not read YAML file {config_path}: {e}. Using script's default arguments.")
            return {}
    else:
        print(f"Warning: YAML config file {config_path} not found. Using script's default arguments.")
        return {}

# --- Helper function to get value from config dict or use default (same as in other steps) ---
def get_config_value(config_dict, section_name, key_name, default_value):
    if config_dict and section_name in config_dict and \
       isinstance(config_dict[section_name], dict) and \
       key_name in config_dict[section_name]:
        yaml_value = config_dict[section_name][key_name]
        if yaml_value is None:
            return default_value if default_value is not None else None
        return yaml_value
    return default_value
# --- 导入本地模块 ---
try:
    # 确保 dataset.py 是能返回 6D 特征且与训练时一致的版本
    from dataset import ShapeNetPartSegDataset
    # 确保 model.py 是能处理 6D 特征且与训练时一致的版本
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


# --- 可视化函数 (保持不变) ---
def visualize_sample_by_class(features_np_xyz, pred_labels_np, num_classes, sample_idx,
                              label_to_color):  # 参数名修改为 features_np_xyz
    """ 按预测类别逐个显示单个样本的点云 (只使用 XYZ)。"""
    if not OPEN3D_AVAILABLE: print("Open3D not available."); return

    # points_np 已经是 XYZ 了
    points_np = features_np_xyz

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
        label_color = label_to_color[label_id % num_classes]  # 使用 % num_classes 防止索引越界
        pcd_label.paint_uniform_color(label_color)

        window_title = f"Sample {sample_idx} - Predicted Label {label_id} ({points_for_label.shape[0]} points)"
        o3d.visualization.draw_geometries([pcd_label], window_name=window_title, width=800, height=600)
        print(f"  Closed window for Label {label_id}.")

    print(f"--- Finished visualizing sample {sample_idx} ---")


# --- 工具函数 - 计算指标 (保持不变) ---
def calculate_metrics_overall(pred_labels_all, target_labels_all, num_classes):
    total_points = target_labels_all.size
    correct_points = np.sum(pred_labels_all == target_labels_all)
    overall_accuracy = (correct_points / total_points) * 100.0 if total_points > 0 else 0.0
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for cl in range(num_classes):
        pred_inds = (pred_labels_all == cl)
        target_inds = (target_labels_all == cl)
        intersection[cl] = np.logical_and(pred_inds, target_inds).sum()
        union[cl] = np.logical_or(pred_inds, target_inds).sum()
    iou_per_class = np.full(num_classes, np.nan)
    has_union = (union > 0)
    iou_per_class[has_union] = intersection[has_union] / union[has_union]
    mIoU = np.nanmean(iou_per_class) * 100.0 if np.any(~np.isnan(iou_per_class)) else 0.0
    return overall_accuracy, mIoU, iou_per_class * 100.0


# --- 评估函数 ---
def run_evaluation(model, dataloader, device, num_classes, indices_to_visualize, label_to_color, args):
    model.eval()
    all_pred_labels_list = []
    all_target_labels_list = []
    total_processed_points = 0

    print(f"\nEvaluating on '{args.partition}' partition...")
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating on {args.partition}")
        for batch_idx, (features, seg_labels) in pbar:
            features, seg_labels = features.to(device), seg_labels.to(device)
            batch_size = features.shape[0]
            # print(features) # 调试时可以取消注释这行来查看输入特征
            logits = model(features)
            predictions = torch.argmax(logits, dim=2)

            current_preds_flat = predictions.cpu().numpy().flatten()
            current_labels_flat = seg_labels.cpu().numpy().flatten()
            all_pred_labels_list.append(current_preds_flat)
            all_target_labels_list.append(current_labels_flat)
            total_processed_points += len(current_labels_flat)

            if indices_to_visualize:
                for i in range(batch_size):
                    sample_idx_global = batch_idx * dataloader.batch_size + i  # 修正获取 batch_size 的方式
                    if sample_idx_global in indices_to_visualize:
                        print(f"\nVisualizing sample at global index: {sample_idx_global}")
                        sample_features_np_all = features[i].cpu().numpy()  # (N, 6)
                        sample_points_np_xyz = sample_features_np_all[:, :3]  # 只取 XYZ
                        sample_pred_labels_np = predictions[i].cpu().numpy()
                        visualize_sample_by_class(
                            sample_points_np_xyz,  # 传入 XYZ
                            sample_pred_labels_np,
                            num_classes,
                            sample_idx_global,
                            label_to_color
                        )
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

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset from: {args.data_root}")
    try:
        # 确保这里的 num_points 与模型训练时一致
        eval_dataset = ShapeNetPartSegDataset(data_root=args.data_root, partition=args.partition,
                                              num_points=args.num_points_hdf5, runtime_num_points=args.runtime_max_points_eval, augment=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    if len(eval_dataset) == 0:
        print(f"Error: No data loaded for partition '{args.partition}'.")
        return

    # 使用 args.batch_size 作为 dataloader 的 batch_size
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             drop_last=False)
    print(f"Loaded '{args.partition}' dataset with {len(eval_dataset)} samples.")

    print(f"Loading model checkpoint from: {args.checkpoint}")
    if not os.path.isfile(args.checkpoint):  # 检查是否是文件
        print(f"FATAL Error: Checkpoint not found or is not a file: {args.checkpoint}")
        sys.exit(1)
    try:
        # !!! 关键: 这里的 args 必须包含与训练时完全相同的模型架构参数 !!!
        # PyG_PointTransformerSegModel 会从 args 中读取这些参数
        model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
        print(
            "Model structure initialized (expects 6D input). Ensure provided model args match the checkpoint's training args.")

        # 考虑从 checkpoint 中加载 args (如果训练脚本保存了它们)
        # 这会使得评估脚本对参数变化更鲁棒，但当前模型类初始化依赖外部传入的 args
        # 例如:
        # saved_checkpoint_data = torch.load(args.checkpoint, map_location=device)
        # if 'args' in saved_checkpoint_data:
        #     print("Loading model architecture args from checkpoint.")
        #     model_args_from_ckpt = saved_checkpoint_data['args']
        #     # 你可能需要将 model_args_from_ckpt 与当前的 args 合并或选择性使用
        #     # 例如: args.k_neighbors = model_args_from_ckpt.k_neighbors 等
        #     # 这需要你的 PyG_PointTransformerSegModel 能够处理这种 args 对象
        # model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=model_args_from_ckpt_or_merged_args).to(device)

        checkpoint_data = torch.load(args.checkpoint,
                                     map_location=device , weights_only=False) # PyTorch 1.13+ 建议 weights_only=True 如果只加载权重

        # 检查 checkpoint 结构
        if 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
            print("Loaded 'model_state_dict' from checkpoint.")
        elif 'state_dict' in checkpoint_data:  # 有些会用 'state_dict'
            model.load_state_dict(checkpoint_data['state_dict'])
            print("Loaded 'state_dict' from checkpoint.")
        else:
            # 尝试直接加载，假设 checkpoint 就是 model state_dict 本身
            model.load_state_dict(checkpoint_data)
            print("Loaded checkpoint directly (assumed to be model state_dict).")
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    indices_to_visualize = set()
    if args.visualize_n_samples > 0 and OPEN3D_AVAILABLE:  # 再次检查 OPEN3D_AVAILABLE
        if args.visualize_n_samples > len(eval_dataset):
            print(
                f"Warning: visualize_n_samples ({args.visualize_n_samples}) > dataset size ({len(eval_dataset)}). Visualizing all samples.")
            args.visualize_n_samples = len(eval_dataset)
        if len(eval_dataset) > 0:  # 确保数据集非空
            random.seed(args.seed)
            indices_to_visualize = set(random.sample(range(len(eval_dataset)), args.visualize_n_samples))
            print(
                f"Will visualize predictions for {len(indices_to_visualize)} randomly selected samples: {sorted(list(indices_to_visualize))}")
        else:
            print("Dataset is empty, cannot select samples for visualization.")
            args.visualize_n_samples = 0

    # 确保 label_to_color 的数量至少是 num_classes
    np.random.seed(42)  # 固定颜色种子以便颜色一致
    # 创建一个颜色映射表，数量可以比 num_classes 多一些，以防万一
    # 使用更鲜明的颜色
    distinct_colors = [
        [230 / 255, 25 / 255, 75 / 255],  # Red
        [60 / 255, 180 / 255, 75 / 255],  # Green
        [255 / 255, 225 / 255, 25 / 255],  # Yellow
        [0 / 255, 130 / 255, 200 / 255],  # Blue
        [245 / 255, 130 / 255, 48 / 255],  # Orange
        [145 / 255, 30 / 255, 180 / 255],  # Purple
        [70 / 255, 240 / 255, 240 / 255],  # Cyan
        [240 / 255, 50 / 255, 230 / 255],  # Magenta
        [210 / 255, 245 / 255, 60 / 255],  # Lime
        [250 / 255, 190 / 255, 212 / 255],  # Pink
        [0 / 255, 128 / 255, 128 / 255],  # Teal
        [220 / 255, 190 / 255, 255 / 255],  # Lavender
        [170 / 255, 110 / 255, 40 / 255],  # Brown
        [255 / 255, 250 / 255, 200 / 255],  # Beige
        [128 / 255, 0 / 255, 0 / 255],  # Maroon
        [128 / 255, 128 / 255, 0 / 255],  # Olive
        [0 / 255, 0 / 255, 128 / 255],  # Navy
        [128 / 255, 128 / 255, 128 / 255]  # Grey
    ]
    if args.num_classes > len(distinct_colors):
        additional_colors = np.random.rand(args.num_classes - len(distinct_colors), 3)
        label_to_color = np.array(distinct_colors + list(additional_colors))
    else:
        label_to_color = np.array(distinct_colors[:args.num_classes])

    start_eval_time = time.time()
    overall_accuracy, mIoU, iou_per_class = run_evaluation(
        model, eval_loader, device, args.num_classes, indices_to_visualize, label_to_color, args
    )
    end_eval_time = time.time()

    print("\n--- Evaluation Results ---")
    print(f"Partition Evaluated: {args.partition}")
    print(f"Checkpoint Used: {args.checkpoint}")
    print(f"Overall Point Accuracy: {overall_accuracy:.2f}%")
    print(f"Mean IoU (mIoU): {mIoU:.2f}%")
    print("\nIoU per class:")
    for i in range(args.num_classes):
        iou_val = iou_per_class[i]
        print(f"  Class {i:2d}: {iou_val:.2f}%" if not np.isnan(iou_val) else f"  Class {i:2d}: NaN")
    print("-" * 25)
    print(f"Evaluation completed in {end_eval_time - start_eval_time:.2f} seconds.")


# --- 命令行参数解析 ---
if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_file', type=str, default='pose_estimation_config.yaml',
                            help='Path to the YAML configuration file.')
    cli_args, _ = pre_parser.parse_known_args()
    config_data = load_config_from_yaml(cli_args.config_file)

    parser = argparse.ArgumentParser(
        description='Evaluate a trained PyG PointTransformer Segmentation model (XYZRGB input), with YAML config.')
    parser.add_argument('--config_file', type=str, default=cli_args.config_file, help='Path to YAML configuration.')

    # --- Checkpoint, Data, and Model Config (sourced from various YAML sections) ---
    eval_setup_group = parser.add_argument_group('Evaluation Setup (from YAML or CLI)')
    eval_setup_group.add_argument('--checkpoint', type=str, 
                        default=get_config_value(config_data, 'InputOutput', 'checkpoint_semantic', 'checkpoints_seg_tesla_part1_normalized/best_model.pth'),
                        help='Path to the trained model checkpoint (.pth file)')
    eval_setup_group.add_argument('--data_root', type=str, 
                        default=get_config_value(config_data, 'HDF5CreationConfig', 'output_hdf5_dir', './data/testla_part1_h5'),
                        help='Path to the directory containing HDF5 files')
    
    # Determine num_points for evaluation dataset (passed as runtime_num_points to Dataset constructor)
    # Priority: EvaluationConfig.runtime_max_points_eval -> TrainingConfig.runtime_max_points_train -> HDF5CreationConfig.num_points_hdf5
    default_runtime_points_eval = get_config_value(config_data, 'EvaluationConfig', 'runtime_max_points_eval', None)
    if default_runtime_points_eval is None or default_runtime_points_eval <= 0:
        default_runtime_points_eval = get_config_value(config_data, 'TrainingConfig', 'runtime_max_points_train', None)
        if default_runtime_points_eval is None or default_runtime_points_eval <= 0:
            default_runtime_points_eval = get_config_value(config_data, 'HDF5CreationConfig', 'num_points_hdf5', 2048) # Fallback to HDF5 point count
    
    eval_setup_group.add_argument('--runtime_max_points_eval', type=int, default=default_runtime_points_eval,
                        help='Max points per sample for evaluation dataset (runtime downsampling). 0/None uses HDF5 native points after checking training config.')
    # This num_points is what the HDF5 files *contain*, used by dataset if runtime_max_points_eval is not effectively set for downsampling
    eval_setup_group.add_argument('--num_points_hdf5', type=int, 
                                  default=get_config_value(config_data, 'HDF5CreationConfig', 'num_points_hdf5', 20480), 
                                  help="Original number of points in HDF5 files (informational, or used if runtime_max_points_eval isn't set to downsample).")

    eval_setup_group.add_argument('--num_classes', type=int, 
                        default=get_config_value(config_data, 'SemanticModelConfig', 'num_classes', 2),
                        help='Number of segmentation classes (MUST match training)')

    # --- Model Architecture (from SemanticModelConfig) ---
    model_arch_group = parser.add_argument_group('Model Architecture (from YAML or CLI - MUST match checkpoint)')
    model_arch_group.add_argument('--k_neighbors', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'k_neighbors', 16))
    model_arch_group.add_argument('--embed_dim', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'embed_dim', 64))
    model_arch_group.add_argument('--pt_hidden_dim', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'pt_hidden_dim', 128))
    model_arch_group.add_argument('--pt_heads', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'pt_heads', 4))
    model_arch_group.add_argument('--num_transformer_layers', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'num_transformer_layers', 2))
    model_arch_group.add_argument('--dropout', type=float, default=get_config_value(config_data, 'SemanticModelConfig', 'dropout', 0.3))

    # --- Evaluation Process (from EvaluationConfig and ControlVisualization) ---
    eval_proc_group = parser.add_argument_group('Evaluation Process (from YAML or CLI)')
    eval_proc_group.add_argument('--batch_size', type=int, 
                        default=get_config_value(config_data, 'EvaluationConfig', 'batch_size_eval', 16),
                        help='Batch size for evaluation')
    eval_proc_group.add_argument('--num_workers', type=int, 
                        default=get_config_value(config_data, 'EvaluationConfig', 'num_dataloader_workers_eval', 4),
                        help='Dataloader workers')
    eval_proc_group.add_argument('--partition', type=str, 
                        default=get_config_value(config_data, 'EvaluationConfig', 'partition_to_eval', 'test'), 
                        choices=['train', 'val', 'test'], help="Which partition to evaluate")
    eval_proc_group.add_argument('--no_cuda', action='store_true', 
                        default=get_config_value(config_data, 'ControlVisualization', 'no_cuda', False),
                        help='Disable CUDA evaluation')
    eval_proc_group.add_argument('--seed', type=int, 
                        default=get_config_value(config_data, 'EvaluationConfig', 'eval_seed', 42),
                        help='Random seed for selecting visualization samples')

    # --- Visualization (from EvaluationConfig) ---
    viz_group = parser.add_argument_group('Visualization (from YAML or CLI)')
    viz_group.add_argument('--visualize_n_samples', type=int, 
                        default=get_config_value(config_data, 'EvaluationConfig', 'visualize_n_samples_eval', 0), metavar='N',
                        help='Randomly select N samples to visualize (0 to disable)')

    args = parser.parse_args(sys.argv[1:])
    main(args)

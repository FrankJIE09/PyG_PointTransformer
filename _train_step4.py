# _train_step4.py
# 版本: 适配加载 6D 特征 (XYZ+RGB) 的 HDF5 数据集进行分割训练
# 注意: 无需因 dataset.py 中的数据归一化而修改此文件。
# dataset.py 现在会自动处理XYZ和RGB的归一化。
# 但训练动态可能会改变，请留意学习率、batch_size等超参数的调整。

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
import datetime
import glob
import yaml
import sys

from dataset import ShapeNetPartSegDataset  # dataset.py 现在包含归一化逻辑
from model import PyG_PointTransformerSegModel


# --- 工具函数 - 计算指标 (保持不变) ---
def calculate_accuracy(pred_logits, target_labels):
    pred_labels = torch.argmax(pred_logits, dim=-1)
    correct = (pred_labels == target_labels).sum()
    total = target_labels.numel()
    if total == 0: return 0.0
    return (correct.item() / total) * 100


def calculate_iou(pred_logits, target_labels, num_classes):
    pred_labels = torch.argmax(pred_logits, dim=-1).cpu().numpy().flatten()
    target_labels = target_labels.cpu().numpy().flatten()
    part_ious = []
    for part_id in range(num_classes):
        pred_inds = (pred_labels == part_id)
        target_inds = (target_labels == part_id)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            iou = np.nan
        else:
            iou = intersection / float(union)
        part_ious.append(iou)
    valid_ious = np.array(part_ious)
    valid_ious = valid_ious[~np.isnan(valid_ious)]
    mIoU = np.mean(valid_ious) * 100.0 if len(valid_ious) > 0 else 0.0
    return mIoU, part_ious * 100


# --- 训练函数 (适配 6D 特征) ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, num_classes, args):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    batches_processed_for_miou = 0  # 更清晰的命名

    pbar = tqdm(dataloader, desc=f"Training Epoch {args.current_epoch}/{args.epochs}")

    for features, seg_labels in pbar:  # features 现在是归一化后的
        features, seg_labels = features.to(device), seg_labels.to(device)

        optimizer.zero_grad()
        logits = model(features)

        loss = criterion(logits.view(-1, num_classes), seg_labels.view(-1).long())
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_acc = calculate_accuracy(logits.view(-1, num_classes), seg_labels.view(-1))
        # 使用 logits.detach() 来计算指标，避免不必要的梯度跟踪
        batch_miou, _ = calculate_iou(logits.detach().view(-1, num_classes), seg_labels.view(-1), num_classes)

        total_loss += batch_loss
        total_acc += batch_acc

        if not np.isnan(batch_miou):
            total_miou += batch_miou
            batches_processed_for_miou += 1
            pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU=f"{batch_miou:.2f}%")
        else:
            pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU="NaN")
            # print(f"Warning: NaN mIoU encountered in training batch. Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%")

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    avg_miou = total_miou / batches_processed_for_miou if batches_processed_for_miou > 0 else 0.0
    return avg_loss, avg_acc, avg_miou


# --- 评估函数 (适配 6D 特征) ---
def evaluate(model, dataloader, criterion, device, num_classes, args):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    batches_processed_for_miou = 0  # 更清晰的命名

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation Epoch {args.current_epoch}/{args.epochs}")
        for features, seg_labels in pbar:  # features 现在是归一化后的
            features, seg_labels = features.to(device), seg_labels.to(device)
            logits = model(features)

            loss = criterion(logits.view(-1, num_classes), seg_labels.view(-1).long())

            batch_loss = loss.item()
            batch_acc = calculate_accuracy(logits.view(-1, num_classes), seg_labels.view(-1))
            batch_miou, _ = calculate_iou(logits.view(-1, num_classes), seg_labels.view(-1), num_classes)

            total_loss += batch_loss
            total_acc += batch_acc

            if not np.isnan(batch_miou):
                total_miou += batch_miou
                batches_processed_for_miou += 1
                pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU=f"{batch_miou:.2f}%")
            else:
                pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU="NaN")
                # print(f"Warning: NaN mIoU encountered in validation batch. Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%")

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    avg_miou = total_miou / batches_processed_for_miou if batches_processed_for_miou > 0 else 0.0
    return avg_loss, avg_acc, avg_miou


# --- 主函数 ---
def main(args):
    start_time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Starting run at: {start_time_str}")
    print(f"Arguments: {args}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_time = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':  # 更安全的设备类型检查
        torch.cuda.manual_seed_all(args.seed)

    print("Loading datasets (XYZ and RGB will be normalized by dataset class)...")
    val_partition = 'val' if glob.glob(os.path.join(args.data_root, 'val*.h5')) else 'test'
    print(f"Validation partition set to: '{val_partition}'")
    train_dataset = ShapeNetPartSegDataset(data_root=args.data_root, partition='train', num_points=args.num_points,
                                           augment=True, runtime_num_points=args.runtime_max_points_train)
    val_dataset = ShapeNetPartSegDataset(data_root=args.data_root, partition=val_partition, num_points=args.num_points,
                                         augment=False, runtime_num_points=args.runtime_max_points_train)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            drop_last=False, pin_memory=(device.type == 'cuda'))
    print("Datasets loaded.")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"Num classes: {args.num_classes}")

    model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
    print("Model initialized (expects 6D input: normalized XYZ + normalized RGB).")
    print(
        f"Model hyperparameters: k={args.k_neighbors}, embed={args.embed_dim}, pt_hidden={args.pt_hidden_dim}, heads={args.pt_heads}, dropout={args.dropout}, layers={args.num_transformer_layers}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    # 考虑是否需要类别权重，如果类别不平衡严重
    # example_weights = torch.tensor([...], device=device)
    # criterion = nn.CrossEntropyLoss(weight=example_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Using ReduceLROnPlateau scheduler, monitoring validation mIoU.")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max', factor=0.5, patience=args.scheduler_patience,  # 使用args控制patience
        threshold=args.scheduler_threshold, threshold_mode='abs', min_lr=1e-7  # 使用args控制threshold
    )

    start_epoch = 0
    best_val_miou = -1.0
    if args.resume:
        best_ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        latest_ckpt_path_glob = os.path.join(args.checkpoint_dir, 'checkpoint_epoch_*.pth')
        list_of_files = glob.glob(latest_ckpt_path_glob)
        latest_ckpt_path = max(list_of_files, key=os.path.getctime, default=None) if list_of_files else None

        ckpt_path_to_load = None
        if args.resume_path and os.path.isfile(args.resume_path):  # 优先使用指定的 resume_path
            ckpt_path_to_load = args.resume_path
        elif os.path.exists(best_ckpt_path):  # 其次尝试 best_model.pth
            ckpt_path_to_load = best_ckpt_path
        elif latest_ckpt_path:  # 最后尝试最新的 checkpoint
            ckpt_path_to_load = latest_ckpt_path

        if ckpt_path_to_load:
            print(f"Resuming from checkpoint: {ckpt_path_to_load}")
            try:
                checkpoint = torch.load(ckpt_path_to_load, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print("Loaded scheduler state.")
                    except Exception as scheduler_load_error:
                        print(
                            f"Could not load scheduler state (may be due to type mismatch or corruption): {scheduler_load_error}. Initializing scheduler.")
                else:
                    print("Scheduler state not found or empty in checkpoint, initializing scheduler.")
                start_epoch = checkpoint.get('epoch', 0)  # 使用 get 避免 KeyError
                best_val_miou = checkpoint.get('best_val_miou', -1.0)
                print(f"Loaded checkpoint (epoch {start_epoch}, best_miou {best_val_miou:.2f}%)")
            except Exception as e:
                print(f"Error loading state dicts: {e}. Training from scratch.")
                start_epoch = 0
                best_val_miou = -1.0
        else:
            print(f"No checkpoint found to resume from. Training from scratch.")

    print(f"\nStarting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, args.epochs):
        args.current_epoch = epoch + 1
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {args.current_epoch}/{args.epochs} --- LR: {current_lr:.6f}")

        train_loss, train_acc, train_miou = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                                            args.num_classes, args)
        val_loss, val_acc, val_miou = evaluate(model, val_loader, criterion, device, args.num_classes, args)

        if not np.isnan(val_miou):
            scheduler.step(val_miou)
        else:
            print("Warning: Validation mIoU is NaN, scheduler did not step based on metric.")
            # scheduler.step() # 或者不带参数地调用，但这通常意味着满足了patience条件

        print(
            f"Epoch {args.current_epoch} Summary: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, mIoU={train_miou:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, mIoU={val_miou:.2f}%")

        is_best = val_miou > best_val_miou and not np.isnan(val_miou)
        if is_best:
            best_val_miou = val_miou
            print(f"*** New Best Validation mIoU: {best_val_miou:.2f}%. Saving model... ***")
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_miou': best_val_miou,
                'args': args
            }
            torch.save(checkpoint_data, save_path)
        elif np.isnan(val_miou):
            print("Validation mIoU is NaN, not saving best model based on this epoch.")
        # else: # 减少不必要的打印
        # print(f"Validation mIoU ({val_miou:.2f}%) did not improve from best ({best_val_miou:.2f}%).")

        if (epoch + 1) % args.save_freq == 0:
            save_path_latest = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            checkpoint_latest_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_miou': best_val_miou,
                'current_val_miou': val_miou,
                'args': args
            }
            torch.save(checkpoint_latest_data, save_path_latest)
            print(f"Saved periodic checkpoint to {save_path_latest}")

    end_time = time.time()
    print("\nTraining finished.")
    print(f"Total training time: {(end_time - start_time) / 3600:.2f} hours")
    print(f"Best Validation mIoU achieved during run: {best_val_miou:.2f}%")
    # ... (rest of the main function remains the same)


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


# --- 命令行参数解析 ---
if __name__ == "__main__":
    # Initial parser for --config_file
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_file', type=str, default='pose_estimation_config.yaml',
                            help='Path to the YAML configuration file.')
    cli_args, _ = pre_parser.parse_known_args()
    config_data = load_config_from_yaml(cli_args.config_file)

    parser = argparse.ArgumentParser(description='PyG PointTransformer Segmentation Training, with YAML config support')

    parser.add_argument('--config_file', type=str, default=cli_args.config_file, help='Path to YAML configuration.')

    # --- Data and Setup (sourced from HDF5CreationConfig and SemanticModelConfig) ---
    data_group = parser.add_argument_group('Data & Setup (from YAML or CLI)')
    data_group.add_argument('--data_root', type=str, 
                        default=get_config_value(config_data, 'HDF5CreationConfig', 'output_hdf5_dir', './data/testla_part1_h5'),
                        help='Path to Segmentation HDF5 data folder (from HDF5CreationConfig.output_hdf5_dir)')
    data_group.add_argument('--num_points', type=int, 
                        default=get_config_value(config_data, 'HDF5CreationConfig', 'num_points_hdf5', 2048),
                        help='Number of points per object IN HDF5 (from HDF5CreationConfig.num_points_hdf5). This is the max points loaded.')
    data_group.add_argument('--num_classes', type=int, 
                        default=get_config_value(config_data, 'SemanticModelConfig', 'num_classes', 2),
                        help='Number of segmentation classes (from SemanticModelConfig.num_classes)')
    data_group.add_argument('--num_workers', type=int, 
                        default=get_config_value(config_data, 'TrainingConfig', 'num_dataloader_workers', 4),
                        help='Dataloader workers (from TrainingConfig.num_dataloader_workers)')
    data_group.add_argument('--runtime_max_points_train', type=int, 
                        default=get_config_value(config_data, 'TrainingConfig', 'runtime_max_points_train', 0),
                        help='Max points to use per sample during training/validation by runtime downsampling (0 or less to disable and use all points from HDF5).')

    # --- Training Process (sourced from TrainingConfig and ControlVisualization) ---
    train_proc_group = parser.add_argument_group('Training Process (from YAML or CLI)')
    train_proc_group.add_argument('--epochs', type=int, 
                        default=get_config_value(config_data, 'TrainingConfig', 'epochs', 500),
                        help='Number of training epochs')
    train_proc_group.add_argument('--batch_size', type=int, 
                        default=get_config_value(config_data, 'TrainingConfig', 'batch_size', 8),
                        help='Batch size')
    train_proc_group.add_argument('--lr', type=float, 
                        default=get_config_value(config_data, 'TrainingConfig', 'learning_rate', 0.001),
                        help='Initial learning rate')
    train_proc_group.add_argument('--weight_decay', type=float, 
                        default=get_config_value(config_data, 'TrainingConfig', 'weight_decay', 1e-4),
                        help='Weight decay for optimizer')
    train_proc_group.add_argument('--scheduler_patience', type=int, 
                        default=get_config_value(config_data, 'TrainingConfig', 'scheduler_patience', 10),
                        help='Patience for ReduceLROnPlateau scheduler')
    train_proc_group.add_argument('--scheduler_threshold', type=float, 
                        default=get_config_value(config_data, 'TrainingConfig', 'scheduler_threshold', 0.001),
                        help='Threshold for ReduceLROnPlateau scheduler')
    train_proc_group.add_argument('--no_cuda', action='store_true', 
                        default=get_config_value(config_data, 'ControlVisualization', 'no_cuda', False),
                        help='Disable CUDA training (from ControlVisualization.no_cuda)')
    train_proc_group.add_argument('--seed', type=int, 
                        default=get_config_value(config_data, 'TrainingConfig', 'train_seed', 42),
                        help='Random seed for this training run (from TrainingConfig.train_seed)')

    # --- Checkpoints (sourced from InputOutput and TrainingConfig) ---
    ckpt_group = parser.add_argument_group('Checkpoints (from YAML or CLI)')
    ckpt_group.add_argument('--checkpoint_dir', type=str, 
                        default=get_config_value(config_data, 'InputOutput', 'checkpoint_semantic', './checkpoints_seg_tesla_part1_normalized'),
                        help='Directory for saving checkpoints (from InputOutput.checkpoint_semantic)')
    ckpt_group.add_argument('--resume', action='store_true', 
                        default=get_config_value(config_data, 'TrainingConfig', 'resume_training', False),
                        help='Resume from best/latest checkpoint (from TrainingConfig.resume_training)')
    ckpt_group.add_argument('--resume_path', type=str, 
                        default=get_config_value(config_data, 'TrainingConfig', 'resume_checkpoint_path', None),
                        help='Specific checkpoint path to resume from (from TrainingConfig.resume_checkpoint_path)')
    ckpt_group.add_argument('--save_freq', type=int, 
                        default=get_config_value(config_data, 'TrainingConfig', 'save_checkpoint_freq', 10),
                        help='Save checkpoint frequency')

    # --- Model Architecture (sourced from SemanticModelConfig) ---
    model_arch_group = parser.add_argument_group('Model Architecture (from YAML or CLI)')
    model_arch_group.add_argument('--k_neighbors', type=int, 
                        default=get_config_value(config_data, 'SemanticModelConfig', 'k_neighbors', 16),
                        help='k for k-NN graph')
    model_arch_group.add_argument('--embed_dim', type=int, 
                        default=get_config_value(config_data, 'SemanticModelConfig', 'embed_dim', 64),
                        help='Initial embedding dimension')
    model_arch_group.add_argument('--pt_hidden_dim', type=int, 
                        default=get_config_value(config_data, 'SemanticModelConfig', 'pt_hidden_dim', 128),
                        help='Hidden dimension for PointTransformerConv')
    model_arch_group.add_argument('--pt_heads', type=int, 
                        default=get_config_value(config_data, 'SemanticModelConfig', 'pt_heads', 4),
                        help='Number of attention heads')
    model_arch_group.add_argument('--num_transformer_layers', type=int, 
                        default=get_config_value(config_data, 'SemanticModelConfig', 'num_transformer_layers', 2),
                        help='Number of PointTransformerConv layers')
    model_arch_group.add_argument('--dropout', type=float, 
                        default=get_config_value(config_data, 'SemanticModelConfig', 'dropout', 0.3),
                        help='Dropout rate')

    args = parser.parse_args(sys.argv[1:])
    args.current_epoch = 0 # Initialize current_epoch, used in train/eval loops
    main(args)
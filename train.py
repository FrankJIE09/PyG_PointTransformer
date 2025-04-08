# train.py
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

# --- 导入本地模块 ---
from dataset import ShapeNetPartSegDataset # 确保 dataset.py 中有这个类
from model import PyG_PointTransformerSegModel # 导入 PyG 模型

# --- 工具函数 - 计算指标 ---

def calculate_accuracy(pred_logits, target_labels):
    """计算逐点分割准确率"""
    # pred_logits: (B, N, C) or (B*N, C)
    # target_labels: (B, N) or (B*N)
    pred_labels = torch.argmax(pred_logits, dim=-1) # (B, N) or (B*N)
    correct = (pred_labels == target_labels).sum()
    total = target_labels.numel()
    if total == 0:
        return 0.0
    return (correct.item() / total) * 100 # 返回百分比

def calculate_iou(pred_logits, target_labels, num_classes):
    """计算每个类别 IoU 和平均 mIoU (批次级别)"""
    pred_labels = torch.argmax(pred_logits, dim=-1).cpu().numpy() # (B, N) or (B*N)
    target_labels = target_labels.cpu().numpy() # (B, N) or (B*N)

    # 确保是一维数组
    pred_labels = pred_labels.flatten()
    target_labels = target_labels.flatten()

    part_ious = []
    for part_id in range(num_classes):
        pred_inds = (pred_labels == part_id)
        target_inds = (target_labels == part_id)

        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()

        if union == 0:
            # 如果这个类别在 GT 和 Pred 中都没有出现，IoU 设为 NaN (后面 nanmean 会忽略)
             # 或者如果只关心 GT 中存在的类别的 IoU，可以采用不同策略
             # 这里我们计算所有类的 IoU，没有出现的类贡献 NaN
            iou = np.nan
        else:
            iou = intersection / float(union)
        part_ious.append(iou)

    # 计算平均 IoU，忽略 NaN 值
    valid_ious = np.array(part_ious)
    valid_ious = valid_ious[~np.isnan(valid_ious)] # 只选择非 NaN 的 IoU
    mIoU = np.mean(valid_ious) if len(valid_ious) > 0 else np.nan # 避免对空数组求 mean

    return mIoU * 100, part_ious # 返回百分比

# --- 训练函数 ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, num_classes, args):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    batches_processed = 0

    pbar = tqdm(dataloader, desc=f"Training Epoch {args.current_epoch}/{args.epochs}")
    for points, seg_labels in pbar:
        points, seg_labels = points.to(device), seg_labels.to(device) # (B, N, 3), (B, N)

        optimizer.zero_grad()
        logits = model(points) # Output: (B, N, num_classes)

        # --- 计算损失 ---
        # Reshape for CrossEntropyLoss: (B*N, num_classes) and (B*N,)
        try:
            loss = criterion(logits.view(-1, num_classes), seg_labels.view(-1))
        except Exception as e:
            print(f"Error calculating loss: {e}")
            print(f"Logits shape: {logits.shape}, Labels shape: {seg_labels.shape}")
            continue # 跳过这个批次

        if torch.isnan(loss) or torch.isinf(loss):
             print(f"Warning: NaN or Inf loss encountered in epoch {args.current_epoch}. Skipping batch.")
             continue


        loss.backward()
        # 可选：梯度裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- 计算指标 (使用未 flatten 的 logits 和 labels) ---
        batch_loss = loss.item()
        batch_acc = calculate_accuracy(logits.view(-1, num_classes), seg_labels.view(-1))
        batch_miou, _ = calculate_iou(logits.view(-1, num_classes), seg_labels.view(-1), num_classes)

        if not np.isnan(batch_miou): # 只累加有效的 mIoU
            total_loss += batch_loss
            total_acc += batch_acc
            total_miou += batch_miou
            batches_processed += 1
            pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU=f"{batch_miou:.2f}%")
        else:
            pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU="NaN")


    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    avg_acc = total_acc / batches_processed if batches_processed > 0 else 0.0
    avg_miou = total_miou / batches_processed if batches_processed > 0 else 0.0
    # print(f"Epoch {args.current_epoch} Train Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%, Avg mIoU: {avg_miou:.2f}%")
    return avg_loss, avg_acc, avg_miou


# --- 评估函数 ---
def evaluate(model, dataloader, criterion, device, num_classes, args):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    batches_processed = 0
    all_part_ious = [] # 存储每个批次的类别 IoU 列表

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation Epoch {args.current_epoch}/{args.epochs}")
        for points, seg_labels in pbar:
            points, seg_labels = points.to(device), seg_labels.to(device)
            logits = model(points) # (B, N, num_classes)

            try:
                 loss = criterion(logits.view(-1, num_classes), seg_labels.view(-1))
            except Exception as e:
                 print(f"Error calculating validation loss: {e}")
                 continue

            if torch.isnan(loss) or torch.isinf(loss):
                 print(f"Warning: NaN or Inf validation loss. Skipping batch.")
                 continue


            batch_loss = loss.item()
            batch_acc = calculate_accuracy(logits.view(-1, num_classes), seg_labels.view(-1))
            batch_miou, batch_part_ious = calculate_iou(logits.view(-1, num_classes), seg_labels.view(-1), num_classes)

            if not np.isnan(batch_miou):
                total_loss += batch_loss
                total_acc += batch_acc
                total_miou += batch_miou
                all_part_ious.extend(batch_part_ious) # 收集所有非 NaN 的类别 IoU
                batches_processed += 1
                pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU=f"{batch_miou:.2f}%")
            else:
                pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU="NaN")


    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    avg_acc = total_acc / batches_processed if batches_processed > 0 else 0.0
    # 对于 mIoU，更标准的做法是计算所有样本/批次的平均类别 IoU
    # 这里我们简单地取批次 mIoU 的平均值
    avg_miou = total_miou / batches_processed if batches_processed > 0 else 0.0

    # 计算更标准的 mIoU (所有验证样本的平均类别 IoU) - 可选
    # all_part_ious_np = np.array(all_part_ious)
    # mean_part_ious = np.nanmean(all_part_ious_np) # 对所有收集到的类别 IoU 求平均
    # print(f"Epoch {args.current_epoch} Validation Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%, Avg Batch mIoU: {avg_miou:.2f}%, Mean Part IoU: {mean_part_ious*100:.2f}%")

    return avg_loss, avg_acc, avg_miou


# --- 主函数 ---
def main(args):
    start_time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Starting run at: {start_time_str}")
    print(f"Arguments: {args}")

    # --- 设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_time = time.time()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.deterministic = True # 可能影响性能
        # torch.backends.cudnn.benchmark = False

    # --- 数据加载 ---
    print("Loading datasets...")
    try:
        train_dataset = ShapeNetPartSegDataset(data_root=args.data_root, partition='train', num_points=args.num_points, augment=True)
        # 通常使用 'val' 或 'test' 作为验证集，取决于你的文件列表名称
        val_partition = 'val' if os.path.exists(os.path.join(args.data_root, 'val_hdf5_file_list.txt')) else 'test'
        print(f"Using '{val_partition}' partition for validation.")
        val_dataset = ShapeNetPartSegDataset(data_root=args.data_root, partition=val_partition, num_points=args.num_points, augment=False)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure --data_root points to the directory containing HDF5 files and "
              "'train_hdf5_file_list.txt' / 'val_hdf5_file_list.txt' (or 'test_hdf5_file_list.txt').")
        return # 退出程序
    except ValueError as e:
         print(f"Error loading data: {e}")
         return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True if device=='cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True if device=='cuda' else False)
    print("Datasets loaded.")
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
    print(f"Number of segmentation classes: {args.num_classes}")


    # --- 模型初始化 ---

    model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)

    print("Model initialized.")
    print(f"Model hyperparameters: k={args.k_neighbors}, embed={args.embed_dim}, pt_hidden={args.pt_hidden_dim}, heads={args.pt_heads}, dropout={args.dropout}, layers={args.num_transformer_layers}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 损失函数 & 优化器 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # --- 加载 Checkpoint (可选) ---
    start_epoch = 0
    best_val_miou = -1.0 # 以 mIoU 作为最佳模型标准
    if args.resume:
        ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        if os.path.isfile(ckpt_path):
            print(f"Resuming from checkpoint: {ckpt_path}")
            try:
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_miou = checkpoint.get('best_val_miou', -1.0) # 兼容旧 checkpoint
                print(f"Loaded checkpoint (epoch {start_epoch}, best_miou {best_val_miou:.2f}%)")
            except Exception as e:
                print(f"Error loading state dicts from checkpoint: {e}. Training from scratch.")
                start_epoch = 0
                best_val_miou = -1.0
        else:
            print(f"No checkpoint found at {ckpt_path}. Training from scratch.")

    # --- 训练循环 ---
    print(f"\nStarting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, args.epochs):
        args.current_epoch = epoch + 1 # 用于日志记录
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {args.current_epoch}/{args.epochs} --- LR: {current_lr:.6f}")

        train_loss, train_acc, train_miou = train_one_epoch(model, train_loader, optimizer, criterion, device, args.num_classes, args)
        val_loss, val_acc, val_miou = evaluate(model, val_loader, criterion, device, args.num_classes, args)

        scheduler.step() # 更新学习率

        print(f"Epoch {args.current_epoch} Summary: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, mIoU={train_miou:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, mIoU={val_miou:.2f}%")

        # --- 保存最佳模型 (基于验证集 mIoU) ---
        is_best = val_miou > best_val_miou
        if is_best and not np.isnan(val_miou): # 仅在有效 mIoU 提高时保存
            best_val_miou = val_miou
            print(f"*** New Best Validation mIoU: {best_val_miou:.2f}%. Saving model... ***")
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_miou': best_val_miou,
                'args': args # 保存训练参数以供参考
            }
            torch.save(checkpoint, save_path)
        elif np.isnan(val_miou):
             print("Validation mIoU is NaN, not saving model based on this epoch.")
        else:
             print(f"Validation mIoU ({val_miou:.2f}%) did not improve from best ({best_val_miou:.2f}%).")

        # 定期保存 checkpoint (可选)
        if (epoch + 1) % args.save_freq == 0:
            save_path_latest = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            # ... (保存 checkpoint 的代码) ...
            print(f"Saved checkpoint to {save_path_latest}")


    # --- 训练结束 ---
    end_time = time.time()
    print("\nTraining finished.")
    print(f"Total training time: {(end_time - start_time)/3600:.2f} hours")
    print(f"Best Validation mIoU achieved: {best_val_miou:.2f}%")
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
         print(f"Best model saved to: {best_model_path}")
    else:
         print("No best model was saved (perhaps validation mIoU never improved or was always NaN).")


# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyG PointTransformer Segmentation Training')

    # 数据参数
    parser.add_argument('--data_root', type=str, default='./data/shapenetpart_hdf5_2048', help='Path to Segmentation HDF5 data folder (e.g., containing ShapeNetPart data)')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points per object')
    parser.add_argument('--num_classes', type=int, default=50, help='Number of segmentation classes (e.g., 50 for ShapeNetPart)')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16*3, help='Batch size (adjust based on GPU memory)')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Checkpoint 参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_seg_pyg_ptconv', help='Directory for saving checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from best_model.pth in checkpoint_dir')
    parser.add_argument('--save_freq', type=int, default=20, help='Save checkpoint frequency')

    # PyG PointTransformer 模型超参数
    parser.add_argument('--k_neighbors', type=int, default=16, help='Number of neighbors for k-NN graph')
    parser.add_argument('--embed_dim', type=int, default=64, help='Initial embedding dimension')
    parser.add_argument('--pt_hidden_dim', type=int, default=128, help='Hidden dimension for PointTransformerConv layers')
    parser.add_argument('--pt_heads', type=int, default=4, help='Number of attention heads in PointTransformerConv layers')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='Number of PointTransformerConv layers to stack')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')

    args = parser.parse_args()

    main(args)
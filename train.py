# train.py
# 版本: 适配加载 6D 特征 (XYZ+RGB) 的 HDF5 数据集进行分割训练
# 注意: 移除了大部分 try...except 块，可能增加调试难度
# 修改: 使用 ReduceLROnPlateau 动态调整学习率

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
import glob # 需要 glob

# --- 导入本地模块 ---
# 确保 dataset.py 是能够加载 data, rgb, seg 并返回 6D features 的版本
from dataset import ShapeNetPartSegDataset
# 确保 model.py 是能够处理 6D features 输入的版本
from model import PyG_PointTransformerSegModel

# --- 工具函数 - 计算指标 (保持不变) ---
def calculate_accuracy(pred_logits, target_labels):
    """计算逐点分割准确率"""
    pred_labels = torch.argmax(pred_logits, dim=-1)
    correct = (pred_labels == target_labels).sum()
    total = target_labels.numel()
    if total == 0: return 0.0
    return (correct.item() / total) * 100

def calculate_iou(pred_logits, target_labels, num_classes):
    """计算每个类别 IoU 和平均 mIoU (批次级别)"""
    pred_labels = torch.argmax(pred_logits, dim=-1).cpu().numpy().flatten()
    target_labels = target_labels.cpu().numpy().flatten()
    part_ious = []
    for part_id in range(num_classes):
        pred_inds = (pred_labels == part_id)
        target_inds = (target_labels == part_id)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            iou = np.nan # 如果联合为空（即该类别在预测和目标中都不存在），IoU设为NaN
        else:
            iou = intersection / float(union)
        part_ious.append(iou)
    valid_ious = np.array(part_ious)
    # 过滤掉 NaN 值 (对应那些在当前批次预测和目标中都未出现的类别)
    valid_ious = valid_ious[~np.isnan(valid_ious)]
    mIoU = np.mean(valid_ious) * 100.0 if len(valid_ious) > 0 else 0.0 # 仅对有效 IoU 求平均
    return mIoU, part_ious * 100


# --- 训练函数 (适配 6D 特征) ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, num_classes, args):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    batches_processed = 0
    pbar = tqdm(dataloader, desc=f"Training Epoch {args.current_epoch}/{args.epochs}")

    # --- 数据解包 (现在是 features, seg_labels) ---
    for features, seg_labels in pbar:
        features, seg_labels = features.to(device), seg_labels.to(device) # (B, N, 6), (B, N)

        optimizer.zero_grad()
        # --- 模型输入 (使用 6D features) ---
        logits = model(features) # -> (B, N, num_classes)

        # --- 计算损失 (逻辑不变) ---
        # 确保 seg_labels 的数据类型是 LongTensor
        loss = criterion(logits.view(-1, num_classes), seg_labels.view(-1).long())
        # 移除 NaN/Inf 检查，若出现则可能在 backward() 崩溃
        loss.backward()
        optimizer.step()

        # --- 计算指标 (逻辑不变) ---
        batch_loss = loss.item()
        batch_acc = calculate_accuracy(logits.view(-1, num_classes), seg_labels.view(-1))
        batch_miou, _ = calculate_iou(logits.view(-1, num_classes), seg_labels.view(-1), num_classes)

        if not np.isnan(batch_miou):
            total_loss += batch_loss
            total_acc += batch_acc
            total_miou += batch_miou
            batches_processed += 1
            pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU=f"{batch_miou:.2f}%")
        else:
            # 如果 mIoU 是 NaN，仍然记录 loss 和 acc，但不计入 mIoU 平均值
            total_loss += batch_loss
            total_acc += batch_acc
            # batches_processed += 1 # 根据是否要将此批次计入 loss/acc 平均来决定是否增加
            pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU="NaN")
            print(f"Warning: NaN mIoU encountered in training batch. Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%")


    # 使用处理过的批次数来计算平均值
    avg_loss = total_loss / len(dataloader) # 或者 batches_processed，取决于上面的选择
    avg_acc = total_acc / len(dataloader)   # 或者 batches_processed
    avg_miou = total_miou / batches_processed if batches_processed > 0 else 0.0
    return avg_loss, avg_acc, avg_miou


# --- 评估函数 (适配 6D 特征) ---
def evaluate(model, dataloader, criterion, device, num_classes, args):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    batches_processed = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation Epoch {args.current_epoch}/{args.epochs}")
        # --- 数据解包 ---
        for features, seg_labels in pbar:
            features, seg_labels = features.to(device), seg_labels.to(device)
            # --- 模型输入 ---
            logits = model(features) # (B, N, num_classes)

            loss = criterion(logits.view(-1, num_classes), seg_labels.view(-1).long())
            # 移除 NaN/Inf 检查

            batch_loss = loss.item()
            batch_acc = calculate_accuracy(logits.view(-1, num_classes), seg_labels.view(-1))
            batch_miou, _ = calculate_iou(logits.view(-1, num_classes), seg_labels.view(-1), num_classes)

            if not np.isnan(batch_miou):
                total_loss += batch_loss
                total_acc += batch_acc
                total_miou += batch_miou
                batches_processed += 1
                pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU=f"{batch_miou:.2f}%")
            else:
                # 如果 mIoU 是 NaN，仍然记录 loss 和 acc，但不计入 mIoU 平均值
                total_loss += batch_loss
                total_acc += batch_acc
                # batches_processed += 1 # 根据是否要将此批次计入 loss/acc 平均来决定是否增加
                pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%", mIoU="NaN")
                print(f"Warning: NaN mIoU encountered in validation batch. Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%")

    # 使用处理过的批次数来计算平均值
    avg_loss = total_loss / len(dataloader) # 或者 batches_processed，取决于上面的选择
    avg_acc = total_acc / len(dataloader)   # 或者 batches_processed
    avg_miou = total_miou / batches_processed if batches_processed > 0 else 0.0
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
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    print("Loading datasets (expecting 6D features)...")
    # --- 数据加载 (使用修改后的 Dataset) ---
    # 初始化失败会直接崩溃
    val_partition = 'val' if glob.glob(os.path.join(args.data_root, 'val*.h5')) else 'test'
    print(f"Validation partition set to: '{val_partition}'")
    train_dataset = ShapeNetPartSegDataset(data_root=args.data_root, partition='train', num_points=args.num_points, augment=True)
    val_dataset = ShapeNetPartSegDataset(data_root=args.data_root, partition=val_partition, num_points=args.num_points, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True if device=='cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True if device=='cuda' else False)
    print("Datasets loaded.")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"Num classes: {args.num_classes}")

    # --- 模型初始化 (使用修改后的 Model) ---
    # 初始化失败会直接崩溃
    model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
    print("Model initialized (expects 6D input: XYZRGB).")
    print(f"Model hyperparameters: k={args.k_neighbors}, embed={args.embed_dim}, pt_hidden={args.pt_hidden_dim}, heads={args.pt_heads}, dropout={args.dropout}, layers={args.num_transformer_layers}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 损失函数 & 优化器 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ===> 修改1: 使用 ReduceLROnPlateau 替换 CosineAnnealingLR <===
    print("Using ReduceLROnPlateau scheduler, monitoring validation mIoU.")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',      # 监控指标希望越大越好 (mIoU)
        factor=0.5,      # 当指标不再提升时，学习率乘以 0.5
        patience=10,     # 在 10 个 epoch 内 Val mIoU 没有提升就降低学习率
        threshold=0.001, # mIoU 提升小于 0.001 (0.1%) 不算提升
        threshold_mode='abs', # 使用绝对阈值比较
        min_lr=1e-7      # 学习率最低降低到 1e-7
    )
    # ===> 修改结束 <===


    # --- 加载 Checkpoint (可选, 保留 try-except 以方便恢复) ---
    start_epoch = 0
    best_val_miou = -1.0
    if args.resume:
        # 尝试从 best_model.pth 加载，如果不存在则尝试最新的 checkpoint
        best_ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        latest_ckpt_path = max(glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint_epoch_*.pth')), key=os.path.getctime, default=None)
        ckpt_path_to_load = best_ckpt_path if os.path.exists(best_ckpt_path) else latest_ckpt_path

        if ckpt_path_to_load and os.path.isfile(ckpt_path_to_load):
            print(f"Resuming from checkpoint: {ckpt_path_to_load}")
            try: # 保留这里的 try-except
                checkpoint = torch.load(ckpt_path_to_load, map_location=device, weights_only=False) # weights_only=False for optimizer/scheduler state
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # 尝试加载 scheduler 状态，如果类型不匹配或不存在则忽略
                if 'scheduler_state_dict' in checkpoint:
                    try:
                        # 简单的加载，如果类型不匹配会出错
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print("Loaded scheduler state.")
                    except Exception as scheduler_load_error:
                         print(f"Could not load scheduler state (may be due to type mismatch or corruption): {scheduler_load_error}. Initializing scheduler.")
                else:
                    print("Scheduler state not found in checkpoint, initializing scheduler.")

                start_epoch = checkpoint['epoch']
                best_val_miou = checkpoint.get('best_val_miou', -1.0) # 使用 get 避免 key 不存在时出错
                print(f"Loaded checkpoint (epoch {start_epoch}, best_miou {best_val_miou:.2f}%)")
            except Exception as e:
                print(f"Error loading state dicts: {e}. Training from scratch.")
                start_epoch = 0
                best_val_miou = -1.0
        else:
            print(f"No checkpoint found at {args.checkpoint_dir}. Training from scratch.")

    # --- 训练循环 ---
    print(f"\nStarting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, args.epochs):
        args.current_epoch = epoch + 1
        current_lr = optimizer.param_groups[0]['lr'] # 获取当前 LR
        print(f"\n--- Epoch {args.current_epoch}/{args.epochs} --- LR: {current_lr:.6f}")

        train_loss, train_acc, train_miou = train_one_epoch(model, train_loader, optimizer, criterion, device, args.num_classes, args)
        val_loss, val_acc, val_miou = evaluate(model, val_loader, criterion, device, args.num_classes, args)

        # ===> 修改2: 调用 scheduler.step() 并传入监控的指标 <===
        if not np.isnan(val_miou): # 确保 val_miou 有效
            scheduler.step(val_miou)
        else:
            print("Warning: Validation mIoU is NaN, scheduler did not step based on metric.")
            # 可以选择让 scheduler 在指标无效时也 step 一次（不带参数），但这可能不符合预期
            # scheduler.step()
        # ===> 修改结束 <===

        print(f"Epoch {args.current_epoch} Summary: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, mIoU={train_miou:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, mIoU={val_miou:.2f}%")

        # --- 保存模型的逻辑 (保持不变) ---
        is_best = val_miou > best_val_miou
        if is_best and not np.isnan(val_miou):
            best_val_miou = val_miou
            print(f"*** New Best Validation mIoU: {best_val_miou:.2f}%. Saving model... ***")
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            # 保存 checkpoint 时 scheduler_state_dict 会自动保存 ReduceLROnPlateau 的状态
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # 保存 ReduceLROnPlateau 状态
                'best_val_miou': best_val_miou,
                'args': args # 保存命令行参数
            }
            torch.save(checkpoint, save_path)
        elif np.isnan(val_miou):
            print("Validation mIoU is NaN, not saving best model based on this epoch.")
        else:
            print(f"Validation mIoU ({val_miou:.2f}%) did not improve from best ({best_val_miou:.2f}%).")

        # --- 定期保存逻辑 (保持不变) ---
        if (epoch + 1) % args.save_freq == 0:
            save_path_latest = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            checkpoint_latest = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_miou': best_val_miou, # 保存当前已知的最佳 mIoU
                'current_val_miou': val_miou,   # 保存当前 epoch 的 mIoU
                'args': args
            }
            torch.save(checkpoint_latest, save_path_latest)
            print(f"Saved periodic checkpoint to {save_path_latest}")

    end_time = time.time()
    print("\nTraining finished.")
    print(f"Total training time: {(end_time - start_time)/3600:.2f} hours")
    print(f"Best Validation mIoU achieved during run: {best_val_miou:.2f}%")
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print(f"Best model saved to: {best_model_path}")
    else:
        # 如果从未保存过 best_model（例如验证一直 NaN），则提示
        if best_val_miou <= 0:
             print("Warning: No best model was saved during the run (possibly due to consistently low or NaN validation mIoU).")
        else:
             # 这种情况理论上不应发生，除非保存失败
             print("Warning: Best model file not found despite best mIoU being recorded.")


# --- 命令行参数解析 (内容保持不变，仅格式化) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyG PointTransformer Segmentation Training with RGB features')

    # 数据和基础设置参数
    parser.add_argument('--data_root', type=str, default='./data/my_custom_dataset_h5_rgb',
                        help='Path to Segmentation HDF5 data folder (containing data, rgb, seg keys)')
    parser.add_argument('--num_points', type=int, default=40960, help='Number of points per object')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of segmentation classes') # 用户需要根据自己数据修改
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers')

    # 训练过程参数
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs') # 恢复默认epochs
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size') # 恢复默认batch_size
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate') # 恢复默认lr
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Checkpoint 相关参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_seg_pyg_ptconv_rgb',
                        help='Directory for saving checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume from best_model.pth or latest checkpoint in checkpoint_dir')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint frequency')

    # 模型架构参数 (Point Transformer)
    parser.add_argument('--k_neighbors', type=int, default=16, help='(Model Arch) k for k-NN graph')
    parser.add_argument('--embed_dim', type=int, default=64, help='(Model Arch) Initial embedding dimension')
    parser.add_argument('--pt_hidden_dim', type=int, default=128, help='(Model Arch) Hidden dimension for PointTransformerConv')
    parser.add_argument('--pt_heads', type=int, default=4, help='(Model Arch) Number of attention heads')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='(Model Arch) Number of PointTransformerConv layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='(Model Arch) Dropout rate')

    args = parser.parse_args()

    # 为 train_one_epoch 和 evaluate 函数添加 current_epoch 属性
    args.current_epoch = 0

    main(args)
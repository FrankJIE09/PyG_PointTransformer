# create_hdf5_from_txt.py
# 版本: 读取 XYZRGB+Label TXT, 保存 XYZ, RGB, Seg 到 HDF5

import os
import glob
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import random
import sys

def sample_or_pad_data(points, rgb, labels, num_target_points):
    """
    对点云坐标、颜色和标签进行采样或填充，以达到目标点数。
    确保对三者使用相同的索引。
    """
    assert points.shape[0] == labels.shape[0] == rgb.shape[0], \
        f"Points({points.shape[0]}), RGB({rgb.shape[0]}), and labels({labels.shape[0]}) must have the same count"
    current_num_points = points.shape[0]

    if current_num_points == 0: return None, None, None

    if current_num_points == num_target_points:
        choice_idx = np.arange(current_num_points)
    elif current_num_points > num_target_points:
        # 随机下采样
        choice_idx = np.random.choice(current_num_points, num_target_points, replace=False)
    else: # current_num_points < num_target_points
        # 重复上采样
        choice_idx = np.random.choice(current_num_points, num_target_points, replace=True)

    final_points = points[choice_idx, :]
    final_rgb = rgb[choice_idx, :]
    final_labels = labels[choice_idx]

    # 返回正确的类型
    return final_points.astype(np.float32), final_rgb.astype(np.uint8), final_labels.astype(np.int64)

def create_split_hdf5(file_list, output_dir, num_points, batch_size, file_prefix, coord_indices, rgb_indices, label_col):
    """
    处理一个数据子集，生成包含 data, rgb, seg 的 HDF5 文件。
    """
    print(f"\nProcessing split: '{file_prefix}' with {len(file_list)} files...")
    batch_count = 0
    processed_samples_total = 0
    for i in tqdm(range(0, len(file_list), batch_size), desc=f"Creating {file_prefix} HDF5 batches"):
        batch_files = file_list[i : i + batch_size]
        batch_data_list = []
        batch_rgb_list = []
        batch_seg_list = []

        for txt_path in batch_files:
            try:
                # 假设所有列都可以先读为 float
                raw_data = np.loadtxt(txt_path, dtype=np.float32)
            except Exception as e:
                print(f"\nWarning: Error reading TXT file {txt_path}: {e}. Skipping.")
                continue

            # 检查维度和列数
            required_cols = max(max(coord_indices), max(rgb_indices), label_col)
            if raw_data.ndim < 2 or raw_data.shape[1] <= required_cols:
                print(f"\nWarning: Insufficient columns or invalid data in {txt_path}. Shape: {raw_data.shape}, Max required index: {required_cols}. Skipping.")
                continue

            # 提取数据
            try:
                points_np = raw_data[:, coord_indices] # (N_orig, 3)
                # 假设 RGB 范围是 0-255
                rgb_np_float = raw_data[:, rgb_indices] # (N_orig, 3)
                rgb_np = np.clip(rgb_np_float, 0, 255).astype(np.uint8) # 转为 uint8
                labels_np_float = raw_data[:, label_col]      # (N_orig,)
                labels_np = labels_np_float.astype(np.int64) # 转为 int64
            except IndexError:
                 print(f"\nWarning: Invalid column indices for {txt_path}. Shape: {raw_data.shape}. Skipping.")
                 continue
            except ValueError as e:
                 print(f"\nWarning: Error converting columns in {txt_path}. Error: {e}. Skipping.")
                 continue

            if points_np.shape[0] == 0:
                print(f"\nWarning: Empty point cloud in {txt_path}. Skipping.")
                continue

            # 采样或填充 (应用于三者)
            processed_points, processed_rgb, processed_labels = sample_or_pad_data(
                points_np, rgb_np, labels_np, num_points
            )

            if processed_points is not None:
                batch_data_list.append(processed_points)
                batch_rgb_list.append(processed_rgb)
                batch_seg_list.append(processed_labels)

        if not batch_data_list:
            continue

        # 堆叠成批次
        batch_data = np.stack(batch_data_list, axis=0) # (B, num_points, 3)
        batch_rgb = np.stack(batch_rgb_list, axis=0)   # (B, num_points, 3)
        batch_seg = np.stack(batch_seg_list, axis=0)   # (B, num_points)

        # 保存 HDF5 文件
        h5_filename = os.path.join(output_dir, f"{file_prefix}_{batch_count}.h5")
        try:
            with h5py.File(h5_filename, 'w') as f:
                f.create_dataset('data', data=batch_data.astype(np.float32))
                f.create_dataset('rgb', data=batch_rgb.astype(np.uint8)) # 保存为 uint8
                f.create_dataset('seg', data=batch_seg.astype(np.int64))
            processed_samples_total += len(batch_data_list)
        except Exception as e:
             print(f"\nError saving HDF5 file {h5_filename}: {e}")

        batch_count += 1

    print(f"Finished processing split '{file_prefix}'. Processed samples: {processed_samples_total}. Generated HDF5 files: {batch_count}.")
    return batch_count

# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert TXT point clouds (XYZ+RGB+Label) to HDF5 format with data/rgb/seg keys.')

    parser.add_argument('--input_dir', type=str, default='./data/12345678', help='Directory containing the input .txt point cloud files (default: ./data/12345678)')
    parser.add_argument('--output_dir', type=str, default='./data/my_custom_dataset_h5_rgb', help='Directory where the output HDF5 files will be saved (default: ./my_custom_dataset_h5_rgb)') # 修改默认输出目录
    parser.add_argument('--num_points', type=int, default=2048*5, help='Target number of points per sample (default: 2048)')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of samples per HDF5 file (default: 64)')
    parser.add_argument('--train_split', type=float, default=0.7, help='Fraction for training set (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15, help='Fraction for validation set (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling (default: 42)')
    # --- 修改 TXT 文件格式参数 ---
    parser.add_argument('--coord_cols', type=str, default='0,1,2', help='Comma-separated indices for X, Y, Z columns (0-based). Default: "0,1,2".')
    parser.add_argument('--rgb_cols', type=str, default='3,4,5', help='Comma-separated indices for R, G, B columns (0-based). Default: "3,4,5".') # 新增 RGB 列参数
    parser.add_argument('--label_col', type=int, default=6, help='Index of the segmentation label column (0-based). Default: 6.')

    args = parser.parse_args()

    # --- 参数检查 ---
    if not os.path.isdir(args.input_dir): print(f"Error: Input directory not found: {args.input_dir}"); sys.exit(1)
    if args.train_split + args.val_split >= 1.0: print("Error: train_split + val_split must be less than 1.0"); sys.exit(1)
    try:
        coord_indices = [int(i.strip()) for i in args.coord_cols.split(',')]
        if len(coord_indices) != 3: raise ValueError("--coord_cols needs 3 indices")
        rgb_indices = [int(i.strip()) for i in args.rgb_cols.split(',')] # 解析 RGB 索引
        if len(rgb_indices) != 3: raise ValueError("--rgb_cols needs 3 indices")
        if args.label_col < 0: raise ValueError("--label_col must be non-negative")
    except ValueError as e: print(f"Error parsing column indices: {e}"); sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output HDF5 files (with data, rgb, seg keys) will be saved in: {os.path.abspath(args.output_dir)}")

    # --- 查找、打乱、划分文件 (逻辑不变) ---
    txt_files = sorted(glob.glob(os.path.join(args.input_dir, '*.txt')))
    if not txt_files: print(f"Error: No .txt files found in {args.input_dir}"); sys.exit(1)
    print(f"Found {len(txt_files)} total .txt files.")
    random.seed(args.seed)
    random.shuffle(txt_files)
    num_total = len(txt_files); num_train = int(num_total * args.train_split); num_val = int(num_total * args.val_split); num_test = num_total - num_train - num_val
    train_files = txt_files[:num_train]; val_files = txt_files[num_train : num_train + num_val]; test_files = txt_files[num_train + num_val :]
    print(f"Splitting data: {num_train} train, {num_val} validation, {num_test} test files.")

    # --- 分别处理每个子集 (调用修改后的函数) ---
    total_h5_files = 0
    if train_files:
        count = create_split_hdf5(train_files, args.output_dir, args.num_points, args.batch_size, "train", coord_indices, rgb_indices, args.label_col)
        total_h5_files += count
    if val_files:
        count = create_split_hdf5(val_files, args.output_dir, args.num_points, args.batch_size, "val", coord_indices, rgb_indices, args.label_col)
        total_h5_files += count
    if test_files:
        count = create_split_hdf5(test_files, args.output_dir, args.num_points, args.batch_size, "test", coord_indices, rgb_indices, args.label_col)
        total_h5_files += count

    print(f"\nTotal HDF5 files generated across all splits: {total_h5_files}")
    print(f"HDF5 files saved in: {os.path.abspath(args.output_dir)}")
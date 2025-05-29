# _create_hdf5_from_txt_step2.py
# 版本: 自动检测TXT列数，支持XYZRGB+Label或XYZRGB

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
    # 标签可以为 None，如果文件没有标签列
    if labels is not None:
        assert points.shape[0] == labels.shape[0] == rgb.shape[0], \
            f"Points({points.shape[0]}), RGB({rgb.shape[0]}), and labels({labels.shape[0]}) must have the same count"
    else:  # If no labels, just check points and RGB
        assert points.shape[0] == rgb.shape[0], \
            f"Points({points.shape[0]}), and RGB({rgb.shape[0]}) must have the same count"

    current_num_points = points.shape[0]

    if current_num_points == 0: return None, None, None

    if current_num_points == num_target_points:
        choice_idx = np.arange(current_num_points)
    elif current_num_points > num_target_points:
        # 随机下采样
        choice_idx = np.random.choice(current_num_points, num_target_points, replace=False)
    else:  # current_num_points < num_target_points
        # 重复上采样
        choice_idx = np.random.choice(current_num_points, num_target_points, replace=True)

    final_points = points[choice_idx, :]
    final_rgb = rgb[choice_idx, :]

    if labels is not None:
        final_labels = labels[choice_idx]
    else:
        # 如果没有标签，创建一个全零的虚拟标签数组
        final_labels = np.zeros(num_target_points, dtype=np.int64)

    # 返回正确的类型
    return final_points.astype(np.float32), final_rgb.astype(np.uint8), final_labels.astype(np.int64)


def create_split_hdf5(file_list, output_dir, num_points, batch_size, file_prefix, coord_indices, rgb_indices,
                      label_col_explicit, has_label):
    """
    处理一个数据子集，生成包含 data, rgb, seg 的 HDF5 文件。
    label_col_explicit: 用户明确指定的标签列索引，如果用户没有指定，则为 None。
    has_label: 自动检测或用户指定是否包含标签列。
    """
    print(f"\nProcessing split: '{file_prefix}' with {len(file_list)} files...")
    batch_count = 0
    processed_samples_total = 0

    # 动态获取列数和最终的标签列索引 (只对第一个文件进行一次检测)
    # 假设所有文件的列数是相同的
    first_file_path = file_list[0]
    try:
        temp_data = np.loadtxt(first_file_path, dtype=np.float32)
        if temp_data.ndim == 1:
            num_cols_in_file = temp_data.shape[0]
        else:
            num_cols_in_file = temp_data.shape[1]
        print(f"Detected {num_cols_in_file} columns in the first file: {first_file_path}")
    except Exception as e:
        print(f"Error: Could not read the first file {first_file_path} to determine column count: {e}")
        return 0  # 无法继续处理

    # 确定实际使用的标签列索引
    actual_label_col = None
    if has_label:  # 脚本确定应该有标签列
        if label_col_explicit is not None:  # 用户明确指定了标签列
            actual_label_col = label_col_explicit
        else:  # 用户未明确指定标签列，但脚本判断应该有标签列
            # 假设标签列在RGB列之后
            max_rgb_idx = max(rgb_indices)
            if max_rgb_idx + 1 < num_cols_in_file:  # 确保存在下一列
                actual_label_col = max_rgb_idx + 1
                print(f"Automatically inferred label column index: {actual_label_col}")
            else:
                # 这种情况表示根据RGB列后推断，文件列数不够放置标签列
                print(
                    f"Warning: Expected a label column after RGB (index {max_rgb_idx}), but file only has {num_cols_in_file} columns. Proceeding WITHOUT label column for this split.")
                has_label = False  # 此时，强制设为无标签

    if not has_label:
        print(f"Note: This split will be processed WITHOUT a label column.")

    for i in tqdm(range(0, len(file_list), batch_size), desc=f"Creating {file_prefix} HDF5 batches"):
        batch_files = file_list[i: i + batch_size]

        batch_data_list = []
        batch_rgb_list = []
        batch_seg_list = []

        for txt_path in batch_files:
            try:
                raw_data = np.loadtxt(txt_path, dtype=np.float32)
            except Exception as e:
                print(f"\nWarning: Error reading TXT file {txt_path}: {e}. Skipping.")
                continue

            # 检查维度。对于单点文件，ndim 可能是 1。
            if raw_data.ndim == 1:
                raw_data = np.expand_dims(raw_data, axis=0)  # 转换为 (1, N_cols)
            elif raw_data.ndim == 0:
                print(f"\nWarning: Empty or invalid data in {txt_path}. Skipping.")
                continue

            current_file_num_cols = raw_data.shape[1]

            # 验证列索引是否超出当前文件的实际列数
            max_coord_idx = max(coord_indices)
            max_rgb_idx = max(rgb_indices)

            if max_coord_idx >= current_file_num_cols:
                print(
                    f"\nWarning: Coordinate column index {max_coord_idx} is out of bounds for {txt_path} (has {current_file_num_cols} columns). Skipping.")
                continue
            if max_rgb_idx >= current_file_num_cols:
                print(
                    f"\nWarning: RGB column index {max_rgb_idx} is out of bounds for {txt_path} (has {current_file_num_cols} columns). Skipping.")
                continue
            if has_label and actual_label_col >= current_file_num_cols:
                print(
                    f"\nWarning: Label column index {actual_label_col} is out of bounds for {txt_path} (has {current_file_num_cols} columns). Skipping.")
                continue

            # 提取数据
            try:
                points_np = raw_data[:, coord_indices]  # (N_orig, 3)
                rgb_np_float = raw_data[:, rgb_indices]  # (N_orig, 3)
                rgb_np = np.clip(rgb_np_float, 0, 255).astype(np.uint8)  # 转为 uint8

                labels_np = None
                if has_label and actual_label_col is not None:
                    labels_np_float = raw_data[:, actual_label_col]  # (N_orig,)
                    labels_np = labels_np_float.astype(np.int64)  # 转为 int64
                # 如果没有标签，labels_np 将保持为 None，这会在 sample_or_pad_data 中处理

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
            # labels_np 可能是 None
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
        batch_data = np.stack(batch_data_list, axis=0)  # (B, num_points, 3)
        batch_rgb = np.stack(batch_rgb_list, axis=0)  # (B, num_points, 3)
        batch_seg = np.stack(batch_seg_list, axis=0)  # (B, num_points)

        # 保存 HDF5 文件
        h5_filename = os.path.join(output_dir, f"{file_prefix}_{batch_count}.h5")
        try:
            with h5py.File(h5_filename, 'w') as f:
                f.create_dataset('data', data=batch_data.astype(np.float32))
                f.create_dataset('rgb', data=batch_rgb.astype(np.uint8))  # 保存为 uint8
                f.create_dataset('seg', data=batch_seg.astype(np.int64))
            processed_samples_total += len(batch_data_list)
        except Exception as e:
            print(f"\nError saving HDF5 file {h5_filename}: {e}")

        batch_count += 1

    print(
        f"Finished processing split '{file_prefix}'. Processed samples: {processed_samples_total}. Generated HDF5 files: {batch_count}.")
    return batch_count


# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert TXT point clouds (XYZ+RGB+[Optional Label]) to HDF5 format with data/rgb/seg keys, with automatic column detection.')

    parser.add_argument('--input_dir', type=str, default='./data/lizheng',
                        help='Directory containing the input .txt point cloud files.')
    parser.add_argument('--output_dir', type=str, default='./data/my_custom_dataset_h5_rgb',
                        help='Directory where the output HDF5 files will be saved.')
    parser.add_argument('--num_points', type=int, default=2048 * 50, help='Target number of points per sample.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of samples per HDF5 file.')
    parser.add_argument('--train_split', type=float, default=0.7, help='Fraction for training set.')
    parser.add_argument('--val_split', type=float, default=0.15, help='Fraction for validation set.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling.')
    # --- TXT 文件格式参数 ---
    parser.add_argument('--coord_cols', type=str, default='0,1,2',
                        help='Comma-separated indices for X, Y, Z columns (0-based).')
    parser.add_argument('--rgb_cols', type=str, default='3,4,5',
                        help='Comma-separated indices for R, G, B columns (0-based).')
    parser.add_argument('--label_col', type=int, default=None,
                        help='Index of the segmentation label column (0-based). If not specified, the script will attempt to infer it. If --no_label is set, this parameter is ignored.')
    parser.add_argument('--no_label', action='store_true',
                        help='If set, indicates that the input TXT files do NOT contain a label column. A dummy label column of zeros will be created in HDF5.')

    args = parser.parse_args()

    # --- 参数检查 ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    if args.train_split + args.val_split >= 1.0:
        print("Error: train_split + val_split must be less than 1.0")
        sys.exit(1)

    try:
        coord_indices = [int(i.strip()) for i in args.coord_cols.split(',')]
        if len(coord_indices) != 3: raise ValueError("--coord_cols needs 3 indices")
        rgb_indices = [int(i.strip()) for i in args.rgb_cols.split(',')]
        if len(rgb_indices) != 3: raise ValueError("--rgb_cols needs 3 indices")

        # 验证列索引是否有效（非负）
        if any(idx < 0 for idx in coord_indices): raise ValueError("Coordinate column indices must be non-negative.")
        if any(idx < 0 for idx in rgb_indices): raise ValueError("RGB column indices must be non-negative.")
        if args.label_col is not None and args.label_col < 0: raise ValueError(
            "--label_col must be non-negative if specified.")

    except ValueError as e:
        print(f"Error parsing column indices: {e}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output HDF5 files (with data, rgb, seg keys) will be saved in: {os.path.abspath(args.output_dir)}")

    # --- 查找、打乱、划分文件 (逻辑不变) ---
    txt_files = sorted(glob.glob(os.path.join(args.input_dir, '*.txt')))
    if not txt_files:
        print(f"Error: No .txt files found in {args.input_dir}")
        sys.exit(1)
    print(f"Found {len(txt_files)} total .txt files.")
    random.seed(args.seed)
    random.shuffle(txt_files)
    num_total = len(txt_files)
    num_train = int(num_total * args.train_split)
    num_val = int(num_total * args.val_split)
    num_test = num_total - num_train - num_val
    train_files = txt_files[:num_train]
    val_files = txt_files[num_train: num_train + num_val]
    test_files = txt_files[num_train + num_val:]
    print(f"Splitting data: {num_train} train, {num_val} validation, {num_test} test files.")

    # --- 分别处理每个子集 (调用修改后的函数) ---
    total_h5_files = 0
    # has_label_flag will be determined based on --no_label and inferred label column
    # We pass args.label_col directly, allowing it to be None

    # Initial determination of whether we expect a label column based on user input
    # This will be refined in create_split_hdf5 based on actual file columns
    initial_has_label_expectation = not args.no_label

    if train_files:
        count = create_split_hdf5(train_files, args.output_dir, args.num_points, args.batch_size, "train",
                                  coord_indices, rgb_indices, args.label_col, initial_has_label_expectation)
        total_h5_files += count
    if val_files:
        count = create_split_hdf5(val_files, args.output_dir, args.num_points, args.batch_size, "val", coord_indices,
                                  rgb_indices, args.label_col, initial_has_label_expectation)
        total_h5_files += count
    if test_files:
        count = create_split_hdf5(test_files, args.output_dir, args.num_points, args.batch_size, "test", coord_indices,
                                  rgb_indices, args.label_col, initial_has_label_expectation)
        total_h5_files += count

    print(f"\nTotal HDF5 files generated across all splits: {total_h5_files}")
    print(f"HDF5 files saved in: {os.path.abspath(args.output_dir)}")
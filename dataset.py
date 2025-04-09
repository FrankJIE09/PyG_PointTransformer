# dataset.py
# 版本: 读取含 data, rgb, seg 的 HDF5, 返回 (features, seg)

import os
import glob
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# --- 数据增强函数 (可选，这里只对坐标应用) ---
# 注意：更复杂的颜色增强可能需要额外处理
def rotate_point_cloud(batch_data_xyz): # 只旋转坐标
    # ... (代码同上) ...
    rotated_data = np.zeros(batch_data_xyz.shape, dtype=np.float32)
    for k in range(batch_data_xyz.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle); sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],[0, 1, 0],[-sinval, 0, cosval]])
        shape_pc = batch_data_xyz[k, ...]; rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data_xyz, sigma=0.01, clip=0.05): # 只抖动坐标
    # ... (代码同上) ...
    B, N, C = batch_data_xyz.shape; assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip).astype(np.float32)
    jittered_data += batch_data_xyz
    return jittered_data

def random_scale_point_cloud(batch_data_xyz, scale_low=0.8, scale_high=1.2): # 只缩放坐标
    # ... (代码同上) ...
    B, N, C = batch_data_xyz.shape; scales = np.random.uniform(scale_low, scale_high, B).astype(np.float32)
    scaled_data = np.copy(batch_data_xyz);
    for batch_index in range(B): scaled_data[batch_index,:,:] *= scales[batch_index]
    return scaled_data

# --- 加载 HDF5 分割数据 (修改为加载 rgb) ---
def load_h5_seg_rgb(h5_filename):
    """ 加载 HDF5 文件，假设包含 'data', 'rgb', 'seg' 键。"""
    # 使用 try...except 比较安全，即使之前去掉了，这里加上
    try:
        f = h5py.File(h5_filename, 'r')
        data = f['data'][:] if 'data' in f else None
        rgb = f['rgb'][:] if 'rgb' in f else None
        seg = f['seg'][:] if 'seg' in f else None # 使用 'seg' 键
        f.close()
        # 检查所有键是否存在且形状大致匹配 (忽略点数 N)
        if data is None or rgb is None or seg is None or \
           data.shape[0] != rgb.shape[0] or data.shape[0] != seg.shape[0] or \
           data.shape[1] != rgb.shape[1] or data.shape[1] != seg.shape[1]:
            print(f"Warning: Missing keys ('data', 'rgb', 'seg') or shape mismatch in {h5_filename}. Skipping.")
            return None, None, None
        return data.astype(np.float32), rgb.astype(np.uint8), seg.astype(np.int64)
    except Exception as e:
        print(f"Error loading {h5_filename}: {e}")
        return None, None, None


# --- ShapeNetPart 分割数据集类 (修改版 - 处理 RGB) ---
class ShapeNetPartSegDataset(Dataset):
    def __init__(self, data_root, partition='train', num_points=2048, augment=False):
        self.data_root = data_root
        if not os.path.isdir(self.data_root): raise FileNotFoundError(f"Data directory not found: {self.data_root}")
        self.num_points = num_points
        self.partition = partition
        self.augment = augment if self.partition == 'train' else False

        print(f"Searching for '{partition}*.h5' files in {self.data_root}...")
        h5_files = sorted(glob.glob(os.path.join(self.data_root, f'{partition}*.h5')))
        if not h5_files: raise FileNotFoundError(f"No '{partition}*.h5' files found in directory: {self.data_root}.")
        print(f"Found {len(h5_files)} HDF5 files for partition '{partition}'.")

        self.data_list = [] # 存储坐标
        self.rgb_list = []  # 存储颜色
        self.seg_list = []  # 存储标签

        print(f"Loading {partition} segmentation data (including RGB)...")
        for h5_path in tqdm(h5_files):
            points_batch, rgb_batch, seg_batch = load_h5_seg_rgb(h5_path)
            if points_batch is not None: # 只要有一个非None就认为有效（内部已检查三者）
                self.data_list.append(points_batch)
                self.rgb_list.append(rgb_batch)
                self.seg_list.append(seg_batch)

        if not self.data_list: raise ValueError(f"No valid data loaded for partition {partition}.")

        self.data = np.concatenate(self.data_list, axis=0) # (TotalShapes, N, 3) float32
        self.rgb = np.concatenate(self.rgb_list, axis=0)    # (TotalShapes, N, 3) uint8
        self.seg = np.concatenate(self.seg_list, axis=0)    # (TotalShapes, N)   int64
        print(f"Loaded {self.data.shape[0]} total shapes for {partition}.")
        print(f"Data shape: {self.data.shape}, RGB shape: {self.rgb.shape}, Seg shape: {self.seg.shape}")
        del self.data_list, self.rgb_list, self.seg_list # 释放内存

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        points = self.data[idx] # (N, 3) float32
        rgb = self.rgb[idx]     # (N, 3) uint8
        seg = self.seg[idx]       # (N,)   int64

        # --- 数据增强 (可选, 仅训练时) ---
        # 注意：只对坐标进行几何增强，颜色通常不做或做特定颜色增强
        if self.augment:
            points_batch = np.expand_dims(points, axis=0)
            points_batch = rotate_point_cloud(points_batch)
            points_batch = random_scale_point_cloud(points_batch)
            points_batch = jitter_point_cloud(points_batch)
            points = points_batch[0]
            # 注意：如果增强改变了点序（例如某些采样方法），需要同步修改 RGB 和 Seg
            # 但我们这里的增强不改变点序

        # --- RGB 归一化 ---
        # 将 uint8 [0, 255] 转换为 float32 [0, 1]
        rgb_normalized = rgb.astype(np.float32) / 255.0

        # --- 合并特征 ---
        # 将 XYZ 和 归一化的 RGB 合并为 6 维特征
        features = np.concatenate((points, rgb_normalized), axis=1) # (N, 6)

        # --- 转换为 Tensor ---
        features_tensor = torch.from_numpy(features).float()
        seg_tensor = torch.from_numpy(seg).long()

        # 返回 6 维特征和标签
        # 形状: (num_points, 6), (num_points,)
        return features_tensor, seg_tensor

# --- 测试代码块 ---
if __name__ == '__main__':
    # !!! 修改为你生成 HDF5 的目录 !!!
    DATA_DIR = './my_custom_dataset_h5_rgb' # 使用新的默认输出目录
    NUM_POINTS = 2048

    print(f"Attempting to load data from: {os.path.abspath(DATA_DIR)}")
    if not os.path.isdir(DATA_DIR): print(f"Error: Data directory '{DATA_DIR}' not found.")
    else:
        try:
            print("\nTesting Train Dataset...")
            train_dataset = ShapeNetPartSegDataset(data_root=DATA_DIR, partition='train', num_points=NUM_POINTS, augment=True)
            print(f"Train dataset size: {len(train_dataset)}")
            if len(train_dataset) > 0:
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
                # 现在 dataloader 返回 features 和 seg_labels
                features, seg_labels = next(iter(train_loader))
                print("Train Batch - Features shape:", features.shape) # 应为 [2, NUM_POINTS, 6]
                print("Train Batch - Labels shape:", seg_labels.shape) # 应为 [2, NUM_POINTS]
            else: print("Train dataset loaded but is empty.")
        except Exception as e: print(f"An error occurred with train dataset: {e}"); import traceback; traceback.print_exc()

        try:
            print("\nTesting Validation Dataset...")
            val_partition = 'val' if glob.glob(os.path.join(DATA_DIR, 'val*.h5')) else 'test'
            print(f"Using partition '{val_partition}' for validation testing.")
            val_dataset = ShapeNetPartSegDataset(data_root=DATA_DIR, partition=val_partition, num_points=NUM_POINTS, augment=False)
            print(f"Validation dataset size: {len(val_dataset)}")
            if len(val_dataset) > 0:
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)
                features, seg_labels = next(iter(val_loader))
                print("Validation Batch - Features shape:", features.shape)
                print("Validation Batch - Labels shape:", seg_labels.shape)
            else: print("Validation dataset loaded but is empty.")
        except Exception as e: print(f"An error occurred with validation dataset: {e}"); import traceback; traceback.print_exc()
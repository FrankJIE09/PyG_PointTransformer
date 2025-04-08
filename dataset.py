# dataset.py
# 版本: 移除 try...except

import os
import glob
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# --- 数据增强函数 ---
def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip).astype(np.float32)
    jittered_data += batch_data
    return jittered_data

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.2):
    """ Randomly scale the point cloud """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B).astype(np.float32)
    scaled_data = np.copy(batch_data)
    for batch_index in range(B):
        scaled_data[batch_index,:,:] *= scales[batch_index]
    return scaled_data

# --- 加载 HDF5 分割数据 (无 try...except, 假设文件和键一定存在) ---
def load_h5_seg(h5_filename):
    """
    加载单个 HDF5 文件，直接访问 'data' 和 'pid' 键。
    如果文件不存在、损坏或缺少键，将直接引发错误。
    """
    f = h5py.File(h5_filename, 'r')
    # 直接访问，如果键不存在会抛出 KeyError
    data = f['data'][:].astype(np.float32)
    seg = f['seg'][:].astype(np.int64)
    f.close()
    return data, seg


# --- ShapeNetPart 分割数据集类 (无 try...except) ---
class ShapeNetPartSegDataset(Dataset):
    def __init__(self, data_root, partition='train', num_points=2048, augment=False):
        self.data_root = data_root
        if not os.path.isdir(self.data_root):
             # 如果目录不存在，这里会触发 FileNotFoundError，但我们不捕获它
             pass # 或者可以直接 raise 错误

        self.num_points = num_points
        self.partition = partition
        self.augment = augment if self.partition == 'train' else False

        # --- 直接查找 HDF5 文件 ---
        print(f"Searching for '{partition}*.h5' files in {self.data_root}...")
        h5_files = sorted(glob.glob(os.path.join(self.data_root, f'{partition}*.h5')))

        if not h5_files:
            # 如果找不到文件，这里会触发 FileNotFoundError，但我们不捕获它
             raise FileNotFoundError(f"No '{partition}*.h5' files found in directory: {self.data_root}.")
        else:
            print(f"Found {len(h5_files)} HDF5 files for partition '{partition}'.")

        self.data = []
        self.seg_labels = []

        print(f"Loading {partition} segmentation data...")
        for h5_path in tqdm(h5_files):
            # load_h5_seg 现在会在出错时直接崩溃
            points_batch, seg_batch = load_h5_seg(h5_path)

            # 这里不再检查 None，因为 load_h5_seg 不会返回 None 了
            # 但仍然检查维度以防万一 (虽然严格来说也不是必须的了)
            if len(points_batch.shape) == 3 and len(seg_batch.shape) == 2:
                self.data.append(points_batch)
                self.seg_labels.append(seg_batch)
            else:
                # 如果需要严格处理，这里应该抛出错误而不是打印警告
                print(f"Warning: Unexpected data shape in {h5_path}. Points: {points_batch.shape}, Seg: {seg_batch.shape}. Skipping for now.")
                # raise ValueError(f"Unexpected data shape in {h5_path}")


        if not self.data:
             # 如果没有加载到任何数据，这里会触发 ValueError，但我们不捕获它
             raise ValueError(f"No valid data loaded for partition {partition}. Check HDF5 files and keys ('data', 'pid').")

        self.data = np.concatenate(self.data, axis=0)
        self.seg_labels = np.concatenate(self.seg_labels, axis=0)
        print(f"Loaded {self.data.shape[0]} total shapes for {partition}.")
        print(f"Data shape before sampling: {self.data.shape}")
        print(f"Labels shape before sampling: {self.seg_labels.shape}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        points = self.data[idx]
        seg = self.seg_labels[idx]

        current_points = points.shape[0]
        if current_points == self.num_points:
             final_points = points
             final_seg = seg
        elif current_points > self.num_points:
            choice_idx = np.random.choice(current_points, self.num_points, replace=False)
            final_points = points[choice_idx, :]
            final_seg = seg[choice_idx]
        else: # current_points < self.num_points
            choice_idx = np.random.choice(current_points, self.num_points, replace=True)
            final_points = points[choice_idx, :]
            final_seg = seg[choice_idx]

        if self.augment:
            points_batch = np.expand_dims(final_points, axis=0)
            points_batch = rotate_point_cloud(points_batch)
            points_batch = random_scale_point_cloud(points_batch)
            points_batch = jitter_point_cloud(points_batch)
            final_points = points_batch[0]

        points_tensor = torch.from_numpy(final_points).float()
        seg_tensor = torch.from_numpy(final_seg).long()

        return points_tensor, seg_tensor

# --- 测试代码块 (无 try...except) ---
if __name__ == '__main__':
    DATA_DIR = './data/shapenetpart_hdf5_2048' # 修改这里指向你的数据目录
    NUM_POINTS = 2048

    print(f"Attempting to load data from: {os.path.abspath(DATA_DIR)}")

    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
    else:
        print("\nTesting Train Dataset...")
        # 如果初始化失败会直接崩溃
        train_dataset = ShapeNetPartSegDataset(data_root=DATA_DIR, partition='train', num_points=NUM_POINTS, augment=True)
        print(f"Train dataset size: {len(train_dataset)}")
        if len(train_dataset) > 0:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
            points, seg_labels = next(iter(train_loader))
            print("Train Batch - Points shape:", points.shape)
            print("Train Batch - Labels shape:", seg_labels.shape)
            print("Sample Labels (first pointcloud, first 10 points):", seg_labels[0, :10])
        else:
            print("Train dataset loaded but is empty.")

        print("\nTesting Validation Dataset...")
        val_partition = 'val' if glob.glob(os.path.join(DATA_DIR, 'val*.h5')) else 'test'
        print(f"Using partition '{val_partition}' for validation testing.")
        # 如果初始化失败会直接崩溃
        val_dataset = ShapeNetPartSegDataset(data_root=DATA_DIR, partition=val_partition, num_points=NUM_POINTS, augment=False)
        print(f"Validation dataset size: {len(val_dataset)}")
        if len(val_dataset) > 0:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)
            points, seg_labels = next(iter(val_loader))
            print("Validation Batch - Points shape:", points.shape)
            print("Validation Batch - Labels shape:", seg_labels.shape)
        else:
             print("Validation dataset loaded but is empty.")
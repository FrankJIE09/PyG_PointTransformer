# dataset.py
# 版本: 读取含 data, rgb, seg 的 HDF5, 返回 (features, seg)
# 新增: XYZ 坐标归一化

import os
import glob
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# --- 数据增强函数 (可选，这里只对坐标应用) ---
# 注意：更复杂的颜色增强可能需要额外处理
def rotate_point_cloud(batch_data_xyz):  # 只旋转坐标
    rotated_data = np.zeros(batch_data_xyz.shape, dtype=np.float32)
    for k in range(batch_data_xyz.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle);
        sinval = np.sin(rotation_angle)
        # Rotation around Y axis
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data_xyz[k, ...];
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data_xyz, sigma=0.01, clip=0.05):  # 只抖动坐标
    B, N, C = batch_data_xyz.shape;
    assert (C == 3)  # Assuming C is 3 for XYZ
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip).astype(np.float32)
    jittered_data += batch_data_xyz
    return jittered_data


def random_scale_point_cloud(batch_data_xyz, scale_low=0.8, scale_high=1.2):  # 只缩放坐标
    B, N, C = batch_data_xyz.shape;
    assert (C == 3)  # Assuming C is 3 for XYZ
    scales = np.random.uniform(scale_low, scale_high, B).astype(np.float32)
    scaled_data = np.copy(batch_data_xyz);
    for batch_index in range(B):
        scaled_data[batch_index, :, :] *= scales[batch_index]
    return scaled_data


# --- !!! 新增: XYZ 坐标归一化函数 !!! ---
def normalize_point_cloud_xyz(points_xyz):
    """
    将XYZ坐标归一化：
    1. 计算质心并中心化点云 (subtract centroid).
    2. 计算点云到原点的最大距离并以此缩放点云 (scale to fit roughly in a unit sphere).
    Args:
        points_xyz (np.ndarray): 输入点云, 形状 (N, 3)
    Returns:
        np.ndarray: 归一化后的点云, 形状 (N, 3)
    """
    if points_xyz.shape[0] == 0:
        return points_xyz  # 处理空点云

    centroid = np.mean(points_xyz, axis=0)
    points_centered = points_xyz - centroid

    max_dist = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
    if max_dist < 1e-6:  # 避免除以零或非常小的值 (如果所有点都重合在质心)
        return points_centered  # 或者返回原始 points_xyz，取决于期望行为

    points_normalized = points_centered / max_dist
    return points_normalized.astype(np.float32)


# --- 结束新增 ---


# --- 加载 HDF5 分割数据 (修改为加载 rgb) ---
def load_h5_seg_rgb(h5_filename):
    """ 加载 HDF5 文件，假设包含 'data', 'rgb', 'seg' 键。"""
    try:
        f = h5py.File(h5_filename, 'r')
        data = f['data'][:] if 'data' in f else None
        rgb = f['rgb'][:] if 'rgb' in f else None
        seg = f['seg'][:] if 'seg' in f else None  # 使用 'seg' 键
        f.close()
        if data is None or rgb is None or seg is None or \
                data.shape[0] != rgb.shape[0] or data.shape[0] != seg.shape[0] or \
                data.shape[1] != rgb.shape[1] or data.shape[1] != seg.shape[1] or \
                data.shape[2] != 3 or rgb.shape[2] != 3:  # 确保XYZ和RGB都是3通道
            print(
                f"Warning: Missing keys ('data', 'rgb', 'seg'), shape mismatch, or incorrect channel count in {h5_filename}. Skipping.")
            return None, None, None
        return data.astype(np.float32), rgb.astype(np.uint8), seg.astype(np.int64)
    except Exception as e:
        print(f"Error loading {h5_filename}: {e}")
        return None, None, None


# --- ShapeNetPart 分割数据集类 (修改版 - 处理 RGB) ---
class ShapeNetPartSegDataset(Dataset):
    def __init__(self, data_root, partition='train', num_points=2048, augment=False, runtime_num_points=None):
        self.data_root = data_root
        if not os.path.isdir(self.data_root): raise FileNotFoundError(f"Data directory not found: {self.data_root}")
        self.num_points = num_points
        self.partition = partition
        self.augment = augment if self.partition == 'train' else False
        self.runtime_num_points = runtime_num_points

        print(f"Searching for '{partition}*.h5' files in {self.data_root}...")
        h5_files = sorted(glob.glob(os.path.join(self.data_root, f'{partition}*.h5')))
        if not h5_files: raise FileNotFoundError(f"No '{partition}*.h5' files found in directory: {self.data_root}.")
        print(f"Found {len(h5_files)} HDF5 files for partition '{partition}'.")

        self.data_list = []
        self.rgb_list = []
        self.seg_list = []

        print(f"Loading {partition} segmentation data (including RGB)...")
        for h5_path in tqdm(h5_files):
            points_batch, rgb_batch, seg_batch = load_h5_seg_rgb(h5_path)
            if points_batch is not None:
                self.data_list.append(points_batch)
                self.rgb_list.append(rgb_batch)
                self.seg_list.append(seg_batch)

        if not self.data_list: raise ValueError(f"No valid data loaded for partition {partition}.")

        self.data = np.concatenate(self.data_list, axis=0)
        self.rgb = np.concatenate(self.rgb_list, axis=0)
        self.seg = np.concatenate(self.seg_list, axis=0)
        print(f"Loaded {self.data.shape[0]} total shapes for {partition}.")
        print(f"Data shape: {self.data.shape}, RGB shape: {self.rgb.shape}, Seg shape: {self.seg.shape}")
        del self.data_list, self.rgb_list, self.seg_list

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        points_xyz_original_loaded = self.data[idx]  # Store initially loaded XYZ
        rgb_original_loaded = self.rgb[idx]      
        seg_original_loaded = self.seg[idx]      
        
        # These will be the working copies, potentially downsampled
        point_set_current = points_xyz_original_loaded.copy() 
        rgb_current = rgb_original_loaded.copy() if rgb_original_loaded is not None else None
        seg_current = seg_original_loaded.copy()
        
        current_points_count = point_set_current.shape[0]

        # Runtime Downsampling Block
        if self.runtime_num_points is not None and self.runtime_num_points > 0 and \
                current_points_count > self.runtime_num_points:
            
            choice_idx = np.random.choice(current_points_count, self.runtime_num_points, replace=False)
            
            point_set_current = point_set_current[choice_idx, :]
            if rgb_current is not None: 
                rgb_current = rgb_current[choice_idx, :]
            seg_current = seg_current[choice_idx]
            # current_points_count = point_set_current.shape[0] # Update current_points_count if needed later

        # 1. Data Augmentation (applied to potentially downsampled XYZ)
        # current_points_xyz is now the (potentially downsampled) point_set_current
        working_xyz_for_aug_and_norm = point_set_current 
        if self.augment:
            # Augmentation functions expect a batch, and operate on XYZ only
            points_batch_for_aug = np.expand_dims(working_xyz_for_aug_and_norm, axis=0) 
            points_batch_for_aug = rotate_point_cloud(points_batch_for_aug)
            points_batch_for_aug = random_scale_point_cloud(points_batch_for_aug)
            points_batch_for_aug = jitter_point_cloud(points_batch_for_aug)
            working_xyz_for_aug_and_norm = points_batch_for_aug[0]  # Back to (N_runtime or N_original, 3)

        # 2. XYZ Coordinate Normalization (applied to augmented or original (potentially downsampled) XYZ)
        normalized_points_xyz = normalize_point_cloud_xyz(working_xyz_for_aug_and_norm)

        # 3. RGB Normalization (applied to potentially downsampled RGB)
        # rgb_current is the (potentially downsampled) RGB data
        if rgb_current is not None:
            rgb_normalized = rgb_current.astype(np.float32) / 255.0
        else: # Handle cases where RGB might be missing (though your load_h5_seg_rgb checks this)
              # Create dummy RGB if necessary, matching the number of points in normalized_points_xyz
            num_current_points = normalized_points_xyz.shape[0]
            rgb_normalized = np.full((num_current_points, 3), 0.5, dtype=np.float32) # Grey color
            # print(f"Warning: RGB data was None for item {idx}, using dummy RGB.")

        # Ensure dimensions still match before concatenation (they should if logic is correct)
        if normalized_points_xyz.shape[0] != rgb_normalized.shape[0]:
            # This should ideally not happen if the above logic is correct.
            # Add robust error handling or debugging here if it does.
            raise ValueError(
                f"Mismatch after processing for item {idx}: XYZ points {normalized_points_xyz.shape[0]}, RGB points {rgb_normalized.shape[0]}. "
                f"Original runtime_num_points: {self.runtime_num_points}, points before choice: {current_points_count if 'current_points_count' in locals() else 'N/A'}"
            )

        # --- Merge Features ---
        features = np.concatenate((normalized_points_xyz, rgb_normalized), axis=1)

        # --- Convert to Tensor ---
        features_tensor = torch.from_numpy(features).float()
        seg_tensor = torch.from_numpy(seg_current).long() # Use (potentially downsampled) seg_current

        return features_tensor, seg_tensor


# --- 测试代码块 ---
if __name__ == '__main__':
    DATA_DIR = './my_custom_dataset_h5_rgb'
    NUM_POINTS = 2048
    print(f"Attempting to load data from: {os.path.abspath(DATA_DIR)}")
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
    else:
        try:
            print("\nTesting Train Dataset...")
            # Test with augmentation
            train_dataset = ShapeNetPartSegDataset(data_root=DATA_DIR, partition='train', num_points=NUM_POINTS,
                                                   augment=True)
            print(f"Train dataset size: {len(train_dataset)}")
            if len(train_dataset) > 0:
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
                features, seg_labels = next(iter(train_loader))
                print("Train Batch - Features shape:", features.shape)
                print("Train Batch - Labels shape:", seg_labels.shape)
                # Check feature range for XYZ (should be roughly within [-1, 1] after normalization)
                print("Train Batch - XYZ (first 3 features) min/max:", torch.min(features[:, :, :3]).item(),
                      torch.max(features[:, :, :3]).item())
                # Check feature range for RGB (should be [0, 1])
                print("Train Batch - RGB (last 3 features) min/max:", torch.min(features[:, :, 3:]).item(),
                      torch.max(features[:, :, 3:]).item())
            else:
                print("Train dataset loaded but is empty.")
        except Exception as e:
            print(f"An error occurred with train dataset: {e}"); import traceback; traceback.print_exc()

        try:
            print("\nTesting Validation Dataset (no augmentation)...")
            val_partition = 'val' if glob.glob(os.path.join(DATA_DIR, 'val*.h5')) else 'test'
            print(f"Using partition '{val_partition}' for validation testing.")
            # Test without augmentation
            val_dataset = ShapeNetPartSegDataset(data_root=DATA_DIR, partition=val_partition, num_points=NUM_POINTS,
                                                 augment=False)
            print(f"Validation dataset size: {len(val_dataset)}")
            if len(val_dataset) > 0:
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)
                features, seg_labels = next(iter(val_loader))
                print("Validation Batch - Features shape:", features.shape)
                print("Validation Batch - Labels shape:", seg_labels.shape)
                print("Validation Batch - XYZ (first 3 features) min/max:", torch.min(features[:, :, :3]).item(),
                      torch.max(features[:, :, :3]).item())
                print("Validation Batch - RGB (last 3 features) min/max:", torch.min(features[:, :, 3:]).item(),
                      torch.max(features[:, :, 3:]).item())
            else:
                print("Validation dataset loaded but is empty.")
        except Exception as e:
            print(f"An error occurred with validation dataset: {e}"); import traceback; traceback.print_exc()
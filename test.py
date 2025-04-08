import h5py
import os
import numpy as np

# --- !!! 修改这里为你实际数据目录下的一个 HDF5 文件路径 !!! ---
h5_file_to_check = './data/shapenetpart_hdf5_2048/train0.h5'
# 或者使用绝对路径:
# h5_file_to_check = '/home/sage/Frank/code/PCT_ModelNet40_Classification/data/shapenetpart_hdf5_2048/train0.h5'
# -------------------------------------------------------------

print(f"Checking file: {h5_file_to_check}")

if os.path.exists(h5_file_to_check):
    try: # 在检查脚本里保留 try...except 是安全的
        with h5py.File(h5_file_to_check, 'r') as f:
            print("\nKeys found at the root level:")
            keys = list(f.keys())
            print(keys)

            print("\nDetails for each key:")
            for key in keys:
                try:
                    dataset = f[key]
                    print(f" - Key: '{key}'")
                    print(f"   Shape: {dataset.shape}")
                    if hasattr(dataset, 'dtype'):
                         print(f"   Data Type: {dataset.dtype}")
                    # 打印少量样本值
                    if np.prod(dataset.shape) < 50:
                         print(f"   Values: {dataset[...]}")
                    else:
                         if len(dataset.shape) > 0:
                             sample_slice = tuple(slice(0, min(s, 2)) for s in dataset.shape)
                             print(f"   Sample Values (up to 2x2x...): \n{dataset[sample_slice]}")
                except Exception as inner_e:
                    print(f" - Error accessing key '{key}': {inner_e}")
    except Exception as e:
        print(f"\nError opening or reading HDF5 file {h5_file_to_check}: {e}")
else:
    print(f"\nError: File not found at the specified path: {h5_file_to_check}")
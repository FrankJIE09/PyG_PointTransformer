# test_hdf5_data.py
# 版本: 修正了可视化参数的判断逻辑

import h5py
import numpy as np
import argparse
import os
import sys

# 尝试导入 Open3D 用于可视化
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: Open3D not found. Visualization will be disabled.")
    OPEN3D_AVAILABLE = False

def visualize_ground_truth(pcd_points_np, pcd_labels_np, window_name="Ground Truth Segmentation"):
    """
    使用 Open3D 可视化点云，并根据真实标签着色。
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available, cannot visualize.")
        return
    if pcd_points_np is None or pcd_labels_np is None or pcd_points_np.size == 0 or pcd_labels_np.size == 0:
        print("Visualization skipped: Invalid points or labels provided.")
        return
    if pcd_points_np.shape[0] != pcd_labels_np.shape[0]:
        print(f"Visualization skipped: Points count ({pcd_points_np.shape[0]}) != Labels count ({pcd_labels_np.shape[0]})")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points_np)

    # 尝试获取类别数量的最大值+1作为颜色映射基准
    # 添加检查确保标签不为空
    if pcd_labels_np.size == 0:
         print("Warning: No labels found for coloring.")
         num_classes_in_sample = 1
    else:
        try:
             num_classes_in_sample = int(np.max(pcd_labels_np)) + 1
        except ValueError: # 处理空标签数组的情况
             print("Warning: Could not determine max label index from empty labels.")
             num_classes_in_sample = 1

    print(f"  Max label index found in sample: {num_classes_in_sample - 1}. Using {num_classes_in_sample} colors.")

    np.random.seed(42)
    colors = np.random.rand(num_classes_in_sample, 3)

    try:
        # 在索引前确保标签有效
        valid_labels_mask = (pcd_labels_np >= 0) & (pcd_labels_np < num_classes_in_sample)
        if not np.all(valid_labels_mask):
             print(f"Warning: Some labels are out of range [0, {num_classes_in_sample - 1}]. Clamping them for coloring.")
             clamped_labels = np.clip(pcd_labels_np, 0, num_classes_in_sample - 1)
             point_colors = colors[clamped_labels]
        else:
             point_colors = colors[pcd_labels_np]
    except IndexError: # 以防万一的额外保护
        print("Error: Label index out of range during color mapping despite checks. Using default color.")
        pcd.paint_uniform_color([0.5, 0.5, 0.5]) # 使用灰色作为后备
        point_colors = None # 标记颜色无效

    if point_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(point_colors)

    print("Displaying point cloud colored by ground truth labels...")
    print("(Close the Open3D window to exit script)")
    o3d.visualization.draw_geometries([pcd], window_name=window_name, width=800, height=600)
    print("Visualization window closed.")


def main(args):
    """主函数，加载、检查并可视化 HDF5 数据"""
    h5_file_path = args.h5_file
    sample_index = args.sample_index

    print(f"Checking HDF5 file: {h5_file_path}")

    # 1. 检查文件是否存在
    if not os.path.exists(h5_file_path):
        print(f"Error: File not found at '{h5_file_path}'")
        return

    # 2. 打开 HDF5 文件并检查键 (保留 try-except)
    try:
        with h5py.File(h5_file_path, 'r') as f:
            print("File opened successfully. Checking keys...")
            if 'data' not in f: print("Error: Key 'data' not found."); return
            data_dataset = f['data']
            print(f" - Found key 'data'. Shape: {data_dataset.shape}, Data Type: {data_dataset.dtype}")
            if 'seg' not in f: print("Error: Key 'seg' not found."); return
            seg_dataset = f['seg']
            print(f" - Found key 'seg'. Shape: {seg_dataset.shape}, Data Type: {seg_dataset.dtype}")

            # 检查形状
            if len(data_dataset.shape) != 3 or len(seg_dataset.shape) != 2 or \
               data_dataset.shape[0] != seg_dataset.shape[0] or \
               data_dataset.shape[1] != seg_dataset.shape[1]:
                print("Error: Shape mismatch between 'data' and 'seg'."); return
            if data_dataset.shape[2] != 3:
                print(f"Error: Expected 'data' to have 3 channels (XYZ), got {data_dataset.shape[2]}."); return

            # 3. 提取样本
            num_samples_in_file = data_dataset.shape[0]; num_points_per_sample = data_dataset.shape[1]
            print(f"File contains {num_samples_in_file} samples, each with {num_points_per_sample} points.")
            if not (0 <= sample_index < num_samples_in_file):
                print(f"Error: Sample index {sample_index} out of bounds."); return

            print(f"\nExtracting sample at index: {sample_index}")
            sample_points = data_dataset[sample_index].astype(np.float64)
            sample_labels = seg_dataset[sample_index].astype(np.int64)

            # 4. 打印信息
            unique_labels, counts = np.unique(sample_labels, return_counts=True)
            print(f"Sample Point Cloud Shape: {sample_points.shape}")
            print(f"Sample Labels Shape: {sample_labels.shape}")
            print(f"Unique labels found in sample {sample_index}: {unique_labels}")
            print("Label counts:")
            for label, count in zip(unique_labels, counts): print(f"  Label {label}: {count} points")

            # 5. 可视化 (!!! 修改判断条件 !!!)
            if not args.no_visualize and OPEN3D_AVAILABLE: # <--- 修改在这里
                visualize_ground_truth(sample_points, sample_labels, window_name=f"Ground Truth - {os.path.basename(h5_file_path)} - Sample {sample_index}")
            elif args.no_visualize:
                print("\nVisualization disabled by --no_visualize flag.")
            else: # OPEN3D_AVAILABLE is False
                 print("\nVisualization skipped because Open3D is not available.")


    except Exception as e:
        print(f"An error occurred while processing the HDF5 file: {e}")
        import traceback
        traceback.print_exc()

# --- 命令行参数解析 (修正 --no_visualize 定义) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check HDF5 point cloud segmentation data and visualize a sample.')

    parser.add_argument('--h5_file', type=str, default="./data//my_custom_dataset_h5/test_0.h5", # 保留了默认值
                        help='Path to the HDF5 file to test (default: ./my_custom_dataset_h5/test_0.h5)')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='Index of the sample within the HDF5 file to visualize (default: 0)')
    # --- !!! 修改点: 移除不正确的 default="True" !!! ---
    parser.add_argument('--no_visualize', action='store_true', # default="True" 已移除
                        help='Disable Open3D visualization (default: False, visualization enabled)')

    args = parser.parse_args()

    # --- 修改启动前的检查逻辑 ---
    # if args.no_visualize:
    #     print("Visualization disabled by user.")
    #     # 不需要强制设置 OPEN3D_AVAILABLE = False，函数内部会检查
    #
    # if not args.no_visualize and not OPEN3D_AVAILABLE:
    #     print("Warning: Visualization requested but Open3D is not available. Run with --no_visualize to only check structure.")
        # 脚本可以继续运行检查部分

    main(args)
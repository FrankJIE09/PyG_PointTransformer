# view_segmentation.py
# 版本: 自动查找并依次显示指定目录下的所有 PLY 文件

import argparse
import os
import sys
import glob  # 导入 glob 用于查找文件

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("FATAL Error: Open3D or NumPy not found.")
    print("Please ensure Open3D and NumPy are installed in your environment:")
    print("pip install open3d numpy")
    sys.exit(1)


def main(args):
    """
    主函数，查找目录下的所有 PLY 文件并依次显示。
    """
    input_directory = args.input_dir
    print(f"Searching for *.ply files in directory: {os.path.abspath(input_directory)}")

    # 检查目录是否存在
    if not os.path.isdir(input_directory):
        print(f"Error: Input directory not found at '{input_directory}'")
        return

    # 使用 glob 查找所有 .ply 文件
    # 使用 sorted() 确保按文件名（通常包含时间戳）顺序显示
    ply_files = sorted(glob.glob(os.path.join(input_directory, '*.ply')))

    if not ply_files:
        print(f"No .ply files found in '{input_directory}'.")
        return

    print(f"Found {len(ply_files)} PLY files. Displaying sequentially...")

    # 循环遍历找到的每个 PLY 文件
    for i, ply_file_path in enumerate(ply_files):
        print(f"\n[{i + 1}/{len(ply_files)}] Loading and displaying: {os.path.basename(ply_file_path)}")

        # 读取 PLY 文件
        try:
            pcd = o3d.io.read_point_cloud(ply_file_path)
        except Exception as e:
            print(f"  Error reading PLY file '{ply_file_path}': {e}")
            continue  # 跳过这个文件，继续下一个

        # 检查点云
        if not pcd.has_points():
            print(f"  Warning: File '{ply_file_path}' contains no points. Skipping display.")
            continue
        if not pcd.has_colors():
            print("  Warning: Point cloud does not contain color information. Displaying with default color.")
            # pcd.paint_uniform_color([0.7, 0.7, 0.7]) # 可选

        # 显示点云 (阻塞，直到窗口关闭)
        window_title = f"[{i + 1}/{len(ply_files)}] {os.path.basename(ply_file_path)}"
        print("  Displaying point cloud. Close the window to view the next file...")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_title,
            width=1024,
            height=768
        )
        print(f"  Closed window for: {os.path.basename(ply_file_path)}")

    print("\nFinished displaying all PLY files.")


# --- 命令行参数解析 (修改) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and sequentially visualize all PLY files in a directory.')

    # 将之前的 --ply_file 改为 --input_dir，设为可选，并提供默认值
    parser.add_argument('--input_dir', type=str, default='./capture_output/',
                        help='Directory containing the PLY files to visualize (default: ./inference_output/)')

    args = parser.parse_args()
    main(args)

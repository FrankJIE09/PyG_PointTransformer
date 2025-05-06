# view_scan_result.py
# pip install open3d

import sys
import open3d as o3d
import numpy as np
def main(path="scan_result.ply"):
    # 允许命令行指定文件名：python view_scan_result.py xxx.ply
    if len(sys.argv) > 1:
        path = sys.argv[1]

    print(f"正在加载点云文件: {path}")
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        print("❌ 文件中没有点云数据，或文件路径不正确。")
        return

    # 输出简单统计信息
    print(pcd)
    print("示例坐标范围 (前 5 个点):")
    print(np.asarray(pcd.points)[:5])

    # 打开交互窗口：鼠标左旋转，右平移，滚轮缩放
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Scan Result Viewer ─ 按 Q 退出",
        width=960,
        height=720,
        point_show_normal=False
    )

if __name__ == "__main__":
    main()

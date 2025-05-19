import numpy as np
import open3d as o3d
import os
import argparse
import random # For generating random colors if needed

def load_pointcloud_txt(filepath):
    """
    从包含 XYZ RGB Label 的 TXT 文件加载点云数据。

    期望文件格式：每行包含 7 列数据，用空格或制表符分隔：
    X Y Z R G B Label
    X, Y, Z: 浮点数 (位置)
    R, G, B: 整数 (0-255 颜色)
    Label: 整数 (类别标签)
    """
    if not os.path.exists(filepath):
        print(f"错误：文件未找到 '{filepath}'")
        # Check if the default path was used and give a more specific message
        if filepath == "./data/lizheng/output.txt":
             print("请确保文件 './data/lizheng/output.txt' 存在，或者在命令行中指定正确的文件路径。")
        return None

    print(f"正在从文件加载点云：'{filepath}'")
    try:
        # 使用 numpy.loadtxt 读取数据，会自动检测分隔符
        # 如果数据量非常大，可以考虑 skiprows, max_rows 等参数分块读取，或使用 pandas.read_csv
        data = np.loadtxt(filepath)

        # 检查列数是否正确
        if data.ndim == 0 or data.size == 0: # Handle empty file case
             print(f"错误：文件 '{filepath}' 是空的或无法解析。")
             return None
        if data.ndim == 1: # Handle a single line file case (loadtxt might return 1D array)
             if data.shape[0] < 7:
                 print(f"错误：文件 '{filepath}' 只有一行且列数不足 ({data.shape[0]} 列)，期望至少 7 列。")
                 return None
             elif data.shape[0] > 7:
                 print(f"警告：文件 '{filepath}' 只有一行但列数多于期望 ({data.shape[0]} 列)，将只使用前 7 列。")
                 data = data[:7]
             data = np.expand_dims(data, axis=0) # Make it 2D (1, 7)


        if data.shape[1] < 7:
            print(f"错误：文件 '{filepath}' 的列数不足 ({data.shape[1]} 列)，期望至少 7 列 (X Y Z R G B Label)。")
            return None
        elif data.shape[1] > 7:
             print(f"警告：文件 '{filepath}' 的列数多于期望 ({data.shape[1]} 列)，将只使用前 7 列。")
             data = data[:, :7]


        print(f"成功加载 {data.shape[0]} 个点。")
        return data

    except ValueError as ve:
         print(f"加载文件时发生值错误，可能是数据格式不匹配 (例如，非数字内容) '{filepath}': {ve}")
         return None
    except Exception as e:
        print(f"加载文件时发生未知错误 '{filepath}': {e}")
        return None

def get_label_colors(unique_labels):
    """
    为每个唯一的 Label 值生成一个颜色映射。
    Label 0 通常被视为背景，映射为黑色。
    """
    label_color_map = {}
    # 定义一组预设的视觉差异较大的颜色 (Open3D 颜色范围 0.0-1.0)
    # 颜色顺序经过打乱，以避免 Label 值顺序和颜色顺序的关联性
    predefined_colors = np.array([
        [1.0, 0.0, 0.0],    # 红色
        [0.0, 1.0, 0.0],    # 绿色
        [0.0, 0.0, 1.0],    # 蓝色
        [1.0, 1.0, 0.0],    # 黄色
        [1.0, 0.0, 1.0],    # 品红色
        [0.0, 1.0, 1.0],    # 青色
        [0.7, 0.7, 0.7],    # 灰色
        [0.5, 0.0, 0.0],    # 深红
        [0.0, 0.5, 0.0],    # 深绿
        [0.0, 0.0, 0.5],    # 深蓝
        [0.5, 0.5, 0.0],    # 橄榄
        [0.5, 0.0, 0.5],    # 紫色
        [0.0, 0.5, 0.5],    # 青蓝
        [1.0, 0.5, 0.0],    # 橙色
        [0.5, 1.0, 0.0],    # 柠檬绿
        [0.0, 0.5, 1.0],    # 天蓝
        [1.0, 0.7, 0.8],    # 粉色
        [0.7, 0.8, 1.0],    # 浅蓝
        [0.8, 1.0, 0.7],    # 浅绿
    ])
    num_predefined = predefined_colors.shape[0]

    # 使用一个固定的种子打乱预设颜色，保证每次运行颜色一致
    np.random.seed(42)
    shuffled_indices = np.random.permutation(num_predefined)
    shuffled_colors = predefined_colors[shuffled_indices]


    color_index = 0
    # 遍历所有 unique_labels，确保每个 Label 都有颜色
    for label in sorted(unique_labels): # 按 Label 值排序，使颜色分配更稳定
        if label == 0:
            label_color_map[label] = [0.0, 0.0, 0.0] # Label 0 映射为黑色
        else:
            # 使用打乱后的预设颜色，循环使用
            label_color_map[label] = shuffled_colors[color_index % num_predefined].tolist()
            color_index += 1

            # 如果预设颜色不够，可以使用随机颜色作为补充（慎用，可能不好区分）
            # if color_index >= num_predefined:
            #     random.seed(label) # 使用 Label 值作为种子生成随机颜色
            #     label_color_map[label] = [random.random(), random.random(), random.random()]

    return label_color_map

def visualize_pointcloud(data, mode='rgb'):
    """
    使用 Open3D 可视化点云数据。

    Args:
        data (np.ndarray): 从 TXT 文件加载的 Nx7 NumPy 数组。
        mode (str): 可视化模式，'rgb' 显示原始颜色，'label' 根据 Label 显示颜色。
    """
    if data is None or data.size == 0:
        print("没有点云数据可供可视化。")
        return

    print(f"正在准备点云数据 ({data.shape[0]} 个点)，使用模式：'{mode}'")

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()

    # 设置点的位置 (X, Y, Z)
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])

    # 设置点的颜色
    if mode == 'rgb':
        print("使用原始 RGB 颜色进行可视化。")
        # 提取 RGB 列，转换为 float 并缩放到 0.0-1.0
        # 确保 RGB 数据类型正确，避免意外
        colors_rgb = data[:, 3:6].astype(np.float64) / 255.0
        # 确保颜色值在 [0, 1] 范围内
        colors_rgb = np.clip(colors_rgb, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    elif mode == 'label':
        print("根据 Label 值使用不同颜色进行可视化。")
        # 提取 Label 列，并找到唯一的 Label 值
        # 确保 Label 数据类型为整数
        labels = data[:, 6].astype(int)
        unique_labels = np.unique(labels)
        print(f"找到的唯一 Label 值：{sorted(unique_labels)}")

        # 获取 Label 颜色映射
        label_color_map = get_label_colors(unique_labels)

        # 根据 Label 分配颜色
        colors_label = np.zeros((data.shape[0], 3), dtype=np.float64)
        for label in unique_labels:
            # 找到所有属于当前 Label 的点的索引
            indices = np.where(labels == label)[0]
            # 从颜色映射中获取对应颜色，并分配给这些点
            if label in label_color_map:
                 colors_label[indices] = label_color_map[label]
            else:
                 # 理论上 get_label_colors 会为所有 unique_labels 生成颜色，这里作为备用
                 print(f"警告：Label {label} 没有对应的颜色映射，使用灰色。")
                 colors_label[indices] = [0.5, 0.5, 0.5] # 默认灰色

        # 确保颜色值在 [0, 1] 范围内
        colors_label = np.clip(colors_label, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors_label)

    else:
        print(f"警告：未知可视化模式 '{mode}'。请使用 'rgb' 或 'label'。默认使用 'rgb'。")
        colors_rgb = data[:, 3:6].astype(np.float64) / 255.0
        colors_rgb = np.clip(colors_rgb, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)


    # 可选：添加坐标系，帮助理解点云方向
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    # geometries_to_draw = [pcd, origin]

    geometries_to_draw = [pcd]

    # 使用 Open3D 可视化
    print("正在打开可视化窗口...")
    # 配置渲染选项，例如点的大小 (point_size)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud Visualization ({mode})",
                      width=800, height=600)
    for geom in geometries_to_draw:
        vis.add_geometry(geom)

    # 可以在这里调整视图或渲染选项
    # render_option = vis.get_render_option()
    # render_option.point_size = 2.0 # 调整点的大小

    # 启动可视化事件循环
    vis.run() # run() 是阻塞的，直到窗口关闭
    vis.destroy_window() # 关闭窗口资源

    print("可视化窗口已关闭。")


def main():
    parser = argparse.ArgumentParser(description='Visualize labeled point cloud from a TXT file.')
    # 修改这里，设置 default 参数
    parser.add_argument('filepath', nargs='?', default="./data/lizheng/output.txt", type=str,
                        help='Path to the input TXT point cloud file (X Y Z R G B Label). Defaults to ./data/lizheng/output.txt if not provided.')
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb', 'label'],
                        help='Visualization mode: "rgb" (original color) or "label" (color by label). Default is "rgb".')

    args = parser.parse_args()

    # 检查 default 文件是否存在，如果不存在且用户没有指定文件，则报错
    if args.filepath == "./data/lizheng/output.txt" and not os.path.exists(args.filepath):
         print(f"错误：默认文件 '{args.filepath}' 未找到。请确保文件存在，或在命令行中指定正确的文件路径。")
         return

    # 加载点云数据
    pointcloud_data = load_pointcloud_txt(args.filepath)

    if pointcloud_data is not None:
        # 可视化点云
        visualize_pointcloud(pointcloud_data, mode=args.mode)
    else:
        print("无法加载点云数据，程序退出。")


if __name__ == "__main__":
    main()
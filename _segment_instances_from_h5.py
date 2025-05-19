# _segment_instances_from_h5.py
# 版本: 从 model.py 导入模型, 删除所有 try...except, 全中文打印信息和注释 (警告：不推荐)

# --- 导入标准库 ---
import argparse
import datetime
import os
import random
import sys
import h5py

# --- 导入核心库 ---
import numpy as np
import torch
import torch.nn as nn # 模型内部可能需要
import torch.nn.functional as F # 模型内部可能需要

# --- 导入 PyTorch Geometric 相关库 ---
# 如果导入失败, 脚本将直接崩溃
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import knn_graph

# --- 导入本地模型定义 ---
# --- !!! 核心: 从外部 model.py 文件导入模型类 !!! ---
# 确保 'model.py' 文件存在且包含 'PyG_PointTransformerSegModel' 类
# 如果导入失败 (文件不存在、类不存在、依赖缺失), 脚本将直接崩溃
from model import PyG_PointTransformerSegModel
print("尝试从 model.py 导入 PyG_PointTransformerSegModel...") # 中文打印

# --- 导入可选库 ---
# 如果导入失败, 脚本将直接崩溃
import open3d as o3d
from sklearn.cluster import DBSCAN
print("成功导入 Open3D 和 scikit-learn。") # 中文打印


# --- 可视化函数: 显示语义分割结果 ---
def visualize_semantic_result(points_np, semantic_labels_np, num_classes, window_name="语义预测结果"): # 窗口标题中文化
    """
    使用 Open3D 可视化语义分割结果 (无错误处理)。
    点云根据预测的语义标签进行着色。

    Args:
        points_np (np.ndarray): 点云坐标 (N, 3)。
        semantic_labels_np (np.ndarray): 每个点的预测语义标签 (N,)。
        num_classes (int): 语义类别的总数, 用于生成颜色映射。
        window_name (str): Open3D 可视化窗口的标题。
    """
    # 基础检查仍然保留
    if points_np is None or semantic_labels_np is None or points_np.size == 0:
        print("警告: 为语义可视化提供了空数据。") # 中文打印
        return
    if points_np.shape[0] != semantic_labels_np.shape[0]:
        print(f"错误: 点 ({points_np.shape[0]}) 和语义标签 ({semantic_labels_np.shape[0]}) 维度不匹配。") # 中文打印
        return
    if num_classes <= 0:
        print(f"错误: 无效的类别数量 ({num_classes}) 用于可视化。") # 中文打印
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])

    np.random.seed(42)
    semantic_label_colors = np.random.rand(num_classes, 3)
    if num_classes >= 1: semantic_label_colors[0] = [0.6, 0.6, 0.6]
    if num_classes >= 2: semantic_label_colors[1] = [1.0, 0.7, 0.0]

    # 直接进行颜色映射
    clamped_labels = np.clip(semantic_labels_np, 0, num_classes - 1)
    point_colors = semantic_label_colors[clamped_labels]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    print(f"\n正在显示组合语义预测结果: {window_name}...") # 中文打印
    print("(请关闭 Open3D 窗口以继续)...") # 中文打印
    # 直接调用可视化
    o3d.visualization.draw_geometries([pcd], window_name=window_name, width=1024, height=768)
    print("语义可视化窗口已关闭。") # 中文打印


# --- 核心评估函数: 执行推理 ---
def run_evaluation(model, features_tensor, device, points_processed_np, num_classes, args):
    """
    执行语义分割推理和可选的可视化 (无错误处理)。
    错误将直接导致脚本崩溃。

    Args:
        model (nn.Module): 已加载权重的 PyTorch 模型。
        features_tensor (torch.Tensor): 预处理后的输入特征张量 (1, N, C), 已放置在目标设备上。
        device (torch.device): 模型和数据所在的设备 ('cuda' 或 'cpu')。
        points_processed_np (np.ndarray): 处理后的点云坐标 (N, 3), 用于可视化。
        num_classes (int): 语义类别的总数。
        args (argparse.Namespace): 包含控制参数的对象 (如 args.visualize_semantic)。

    Returns:
        np.ndarray: 预测的语义标签数组 (N,)。如果过程中发生错误, 函数不会返回而是直接崩溃。
    """
    model.eval() # 设置为评估模式
    print(f"正在设备 {device} 上运行语义分割推理...") # 中文打印

    # 在 no_grad 上下文中执行推理，任何错误将导致崩溃
    with torch.no_grad():
        if features_tensor.device != device:
            features_tensor = features_tensor.to(device)
        logits = model(features_tensor)
        predictions = torch.argmax(logits, dim=2)
        pred_semantic_labels_np = predictions.squeeze(0).cpu().numpy()


    print("推理成功。")

    # --- 可选的语义可视化 ---
    if args.visualize_semantic:
        # 直接调用可视化函数，其内部错误也会导致崩溃
        visualize_semantic_result(
            points_processed_np,
            pred_semantic_labels_np,
            num_classes,
            # 窗口标题中文化
            window_name=f"语义预测结果 - 样本 {args.sample_index}"
        )

    # 如果之前的步骤没有崩溃，则返回结果
    return pred_semantic_labels_np


# --- 可视化函数 (实例+标记) ---
def visualize_instances_with_markers(points_np, instance_labels, centroids, marker_radius,
                                     window_name="实例分割结果"): # 窗口标题中文化
    """
    可视化实例分割结果 (无错误处理)。

    Args:
        points_np (np.ndarray): 点云坐标 (N, 3)。
        instance_labels (np.ndarray): 每个点的实例标签 (N,), -1 表示噪声或未聚类。
        centroids (dict): 包含实例质心的字典 {instance_id: centroid_xyz}。
        marker_radius (float): 用于标记质心的球体的半径。
        window_name (str): Open3D 可视化窗口的标题。
    """
    # 基础检查仍然保留
    if points_np is None or instance_labels is None or points_np.size == 0:
        print("警告: 为实例可视化提供了空数据。") # 中文打印
        return
    if points_np.shape[0] != instance_labels.shape[0]:
        print(f"错误: 点 ({points_np.shape[0]}) 和实例标签 ({instance_labels.shape[0]}) 维度不匹配。") # 中文打印
        return

    geometries = []
    unique_instance_ids = np.unique(instance_labels)
    num_instances = len(unique_instance_ids[unique_instance_ids != -1])
    num_noise = np.sum(instance_labels == -1)
    print(f"  正在可视化 {num_instances} 个找到的实例和 {num_noise} 个噪声/背景点。") # 中文打印

    max_instance_id = int(np.max(instance_labels)) if num_instances > 0 else -1
    num_colors_needed = max_instance_id + 2
    if num_colors_needed < 1: num_colors_needed = 1
    np.random.seed(42)
    instance_colors = np.random.rand(num_colors_needed, 3)
    instance_colors[0] = [0.5, 0.5, 0.5]

    pcd_instances = o3d.geometry.PointCloud()
    pcd_instances.points = o3d.utility.Vector3dVector(points_np)

    # 直接进行颜色映射
    indices = instance_labels + 1
    indices = np.clip(indices, 0, num_colors_needed - 1)
    point_colors = instance_colors[indices]
    pcd_instances.colors = o3d.utility.Vector3dVector(point_colors)
    geometries.append(pcd_instances)

    marker_color = [1.0, 0.0, 0.0]
    if centroids:
        for inst_id, centroid in centroids.items():
            if inst_id != -1:
                if centroid is not None and not np.isnan(centroid).any():
                    marker = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
                    marker.paint_uniform_color(marker_color)
                    marker.translate(centroid)
                    geometries.append(marker)
                else:
                     print(f"警告: 实例 {inst_id} 的质心无效: {centroid}。跳过标记。") # 中文打印

    print("正在显示带质心标记的实例分割结果...") # 中文打印
    print("(请关闭 Open3D 窗口以完成)") # 中文打印
    # 直接调用可视化
    o3d.visualization.draw_geometries(geometries, window_name=window_name, width=1024, height=768)
    print("实例可视化窗口已关闭。") # 中文打印


# --- 主函数 (移除了所有 try...except) ---
def main(args):
    """脚本的主执行函数 (无错误处理)。"""
    # 记录开始时间并打印参数
    start_time = datetime.datetime.now()
    print(f"脚本启动于: {start_time.strftime('%Y-%m-%d_%H-%M-%S')}") # 中文打印
    # 打印当前时间
    print(f"当前系统时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") # 中文打印
    print(f"参数: {vars(args)}") # 中文打印

    # --- 1. 依赖确认 (信息性打印) ---
    if args.do_clustering:
        print("请求进行聚类 (--do_clustering)。") # 中文打印
    if args.visualize_semantic or not args.no_visualize or args.save_results:
        print("请求进行可视化或保存PLY，假设 Open3D 可用。") # 中文打印

    # --- 2. 设备设置 ---
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n使用设备: {device}") # 中文打印

    # --- 3. 创建输出目录 ---
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"输出目录已确保/创建: {args.output_dir}") # 中文打印

    # --- 4. 加载语义分割模型 ---
    print(f"\n正在从以下路径加载语义模型检查点: {args.checkpoint}") # 中文打印
    if not os.path.exists(args.checkpoint):
        print(f"致命错误: 在 {args.checkpoint} 未找到检查点文件") # 中文打印
        sys.exit(1)

    # 直接初始化和加载
    model = PyG_PointTransformerSegModel(
        num_classes=args.num_classes,
        args=args
    ).to(device)
    print("模型结构已初始化。") # 中文打印

    checkpoint = torch.load(args.checkpoint, map_location=device,weights_only=False)

    state_dict_key = None
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint: state_dict_key = 'model_state_dict'
        elif 'state_dict' in checkpoint: state_dict_key = 'state_dict'

    if state_dict_key:
        model.load_state_dict(checkpoint[state_dict_key])
        print(f"已从检查点键加载权重: '{state_dict_key}'。") # 中文打印
    else:
        model.load_state_dict(checkpoint)
        print("已直接从检查点加载权重。") # 中文打印

    print("模型权重加载成功。") # 中文打印
    model.eval()

    # --- 5. 加载 HDF5 数据 ---
    print(f"\n正在从 HDF5 文件加载数据: {args.input_h5}") # 中文打印
    if not os.path.exists(args.input_h5):
        print(f"致命错误: 未找到输入 HDF5 文件: {args.input_h5}") # 中文打印
        sys.exit(1)

    # 直接打开和读取
    with h5py.File(args.input_h5, 'r') as f:
        if 'data' not in f:
            print("致命错误: HDF5 文件必须包含 'data' 数据集。") # 中文打印
            sys.exit(1)

        dset_data = f['data']
        dset_rgb = None
        if not args.no_rgb and 'rgb' in f:
            dset_rgb = f['rgb']
            print("在 HDF5 文件中找到 'rgb' 数据集。") # 中文打印
        elif not args.no_rgb:
             print("警告: --no_rgb 为 False, 但未在 HDF5 文件中找到 'rgb' 数据集。") # 中文打印

        num_samples_in_file = dset_data.shape[0]
        num_points_in_file = dset_data.shape[1]

        if not (0 <= args.sample_index < num_samples_in_file):
            print(f"致命错误: 样本索引 {args.sample_index} 超出范围 (文件包含 {num_samples_in_file} 个样本)。") # 中文打印
            sys.exit(1)

        print(f"正在提取索引为 {args.sample_index} 的样本...") # 中文打印
        points_xyz_np = dset_data[args.sample_index, :, :3].astype(np.float32)
        points_rgb_np = None
        if dset_rgb is not None:
             if dset_rgb.shape[0] == num_samples_in_file and \
                dset_rgb.shape[1] == num_points_in_file and \
                dset_rgb.shape[2] == 3:
                  points_rgb_np = dset_rgb[args.sample_index].astype(np.uint8)
             else:
                  print(f"警告: RGB 数据形状不匹配。将忽略 RGB 数据。") # 中文打印

        if num_points_in_file != args.num_points:
             print(f"信息: 样本中的实际点数 ({num_points_in_file}) 与 --num_points ({args.num_points}) 不同。将使用实际点数。") # 中文打印
             args.num_points = num_points_in_file

    # --- 6. 数据预处理 ---
    print(f"\n正在预处理包含 {args.num_points} 个点的点云样本...") # 中文打印
    points_processed_np = np.copy(points_xyz_np)

    if args.normalize:
        # 直接计算
        centroid = np.mean(points_processed_np, axis=0)
        points_processed_np = points_processed_np - centroid
        max_dist = np.max(np.sqrt(np.sum(points_processed_np**2, axis=1)))
        if max_dist > 1e-6:
            points_processed_np = points_processed_np / max_dist
        print("已应用归一化 (中心化和缩放)。") # 中文打印

    features_list = [points_processed_np]
    if not args.no_rgb:
        if points_rgb_np is not None:
            rgb_normalized = points_rgb_np.astype(np.float32) / 255.0
            features_list.append(rgb_normalized)
            print("已添加归一化的 RGB 特征。") # 中文打印
        else:
            default_colors = np.full((points_processed_np.shape[0], 3), 0.5, dtype=np.float32)
            features_list.append(default_colors)
            print("已添加默认颜色特征。") # 中文打印

    # 直接拼接
    features_np = np.concatenate(features_list, axis=1).astype(np.float32)

    model_input_channels = 3 if args.no_rgb else 6
    if features_np.shape[1] != model_input_channels:
        print(f"致命错误: 最终特征维度 ({features_np.shape[1]}) 与预期模型输入维度 ({model_input_channels}) 不符。") # 中文打印
        sys.exit(1)

    features_tensor = torch.from_numpy(features_np).float().unsqueeze(0).to(device)
    print(f"预处理完成。模型输入张量形状: {features_tensor.shape}") # 中文打印

    # --- 7. 执行语义分割推理 ---
    pred_semantic_labels_np = run_evaluation(
        model=model, features_tensor=features_tensor, device=device,
        points_processed_np=points_processed_np, num_classes=args.num_classes, args=args
    )
    print("已获取语义预测结果。") # 中文打印
    unique_semantic, counts_semantic = np.unique(pred_semantic_labels_np, return_counts=True)
    print(f"  预测的语义标签分布: {dict(zip(unique_semantic, counts_semantic))}") # 中文打印

    # --- 8. DBSCAN 聚类 (可选) ---
    instance_labels_full = np.full_like(pred_semantic_labels_np, -1, dtype=np.int64)
    centroids = {}
    num_instances_found = 0
    if args.do_clustering:
        print(f"\n正在对预测为标签 {args.screw_label_id} 的点执行 DBSCAN 聚类...") # 中文打印
        target_mask = (pred_semantic_labels_np == args.screw_label_id)
        target_points_xyz = points_processed_np[target_mask]

        if target_points_xyz.shape[0] < args.dbscan_min_samples:
            print(f"跳过聚类: 仅找到 {target_points_xyz.shape[0]} 个目标点 (需要至少 {args.dbscan_min_samples} 个)。") # 中文打印
        else:
            print(f"找到 {target_points_xyz.shape[0]} 个目标点。正在运行 DBSCAN...") # 中文打印
            # 直接运行 DBSCAN
            db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
            instance_labels_target_points = db.fit_predict(target_points_xyz)
            instance_labels_full[target_mask] = instance_labels_target_points

            valid_instance_labels = instance_labels_target_points[instance_labels_target_points != -1]
            unique_instances, counts = np.unique(valid_instance_labels, return_counts=True)
            num_instances_found = len(unique_instances)
            num_noise_points_in_target = np.sum(instance_labels_target_points == -1)
            print(f"聚类完成: 找到 {num_instances_found} 个实例, {num_noise_points_in_target} 个噪声点 (在目标标签 {args.screw_label_id} 内)。") # 中文打印

            for i, inst_id in enumerate(unique_instances):
                instance_points = target_points_xyz[instance_labels_target_points == inst_id]
                if instance_points.shape[0] > 0:
                    centroid = np.mean(instance_points, axis=0)
                    centroids[inst_id] = centroid
                    # 中文打印质心信息
                    print(f"  - 实例 {inst_id}: {counts[i]} 个点, 质心=[{centroid[0]:.4f},{centroid[1]:.4f},{centroid[2]:.4f}]")
    else:
        print("\n跳过 DBSCAN 聚类。") # 中文打印
        instance_labels_full.fill(-1)
        centroids = {}
        num_instances_found = 0


    # --- 9. 保存结果 (可选) ---
    if args.save_results:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        print(f"\n正在使用时间戳 {timestamp} 保存结果到 {args.output_dir}...") # 中文打印
        files_saved_list = []

        # 直接保存
        points_filename = os.path.join(args.output_dir, f"points_processed_{timestamp}.npy")
        semantic_filename = os.path.join(args.output_dir, f"semantic_labels_{timestamp}.npy")
        instance_filename = os.path.join(args.output_dir, f"instance_labels_{timestamp}.npy")
        txt_filename = os.path.join(args.output_dir, f"segmentation_full_{timestamp}.txt")

        np.save(points_filename, points_processed_np); files_saved_list.append(os.path.basename(points_filename))
        np.save(semantic_filename, pred_semantic_labels_np); files_saved_list.append(os.path.basename(semantic_filename))
        np.save(instance_filename, instance_labels_full); files_saved_list.append(os.path.basename(instance_filename))

        data_to_save = np.hstack((
            points_processed_np, pred_semantic_labels_np.reshape(-1, 1), instance_labels_full.reshape(-1, 1)
        ))
        np.savetxt(txt_filename, data_to_save, fmt='%.6f %.6f %.6f %d %d', delimiter=' ')
        files_saved_list.append(os.path.basename(txt_filename))

        num_ply_saved = 0
        if args.do_clustering and num_instances_found > 0:
            print("正在保存单独的实例 PLY 文件...") # 中文打印
            max_instance_id = int(np.max(list(centroids.keys()))) if centroids else -1
            num_colors_needed = max_instance_id + 2
            if num_colors_needed < 1: num_colors_needed = 1
            np.random.seed(42); instance_colors = np.random.rand(num_colors_needed, 3)
            instance_colors[0] = [0.5, 0.5, 0.5]

            for inst_id in centroids.keys():
                if inst_id == -1: continue
                mask = (instance_labels_full == inst_id)
                points_for_instance = points_processed_np[mask]
                if points_for_instance.shape[0] > 0:
                    pcd_inst = o3d.geometry.PointCloud()
                    pcd_inst.points = o3d.utility.Vector3dVector(points_for_instance)
                    label_color = instance_colors[inst_id + 1]
                    pcd_inst.paint_uniform_color(label_color)
                    ply_filename = os.path.join(args.output_dir, f"instance_{timestamp}_id_{inst_id}.ply")
                    o3d.io.write_point_cloud(ply_filename, pcd_inst, write_ascii=False)
                    num_ply_saved += 1

            if num_ply_saved > 0: files_saved_list.append(f"{num_ply_saved} 个实例 PLY 文件") # 中文

        print(f"成功保存: {', '.join(files_saved_list)}") # 中文打印

    else:
        print("\n跳过保存结果。") # 中文打印

    # --- 10. 最终可视化 ---
    if not args.no_visualize:
        if args.do_clustering and num_instances_found > 0:
             # 可视化函数标题已中文化
             visualize_instances_with_markers(
                 points_processed_np, instance_labels_full, centroids, args.marker_radius,
                 window_name=f"实例分割结果 - 样本 {args.sample_index}" )
        elif args.do_clustering and num_instances_found == 0:
             print("\n跳过最终实例可视化: 已启用聚类, 但未找到实例。") # 中文打印
        elif not args.do_clustering:
             print("\n跳过最终实例可视化: 未启用聚类。") # 中文打印
    else:
        print("\n跳过最终可视化。") # 中文打印

    # 记录结束时间并打印总耗时
    end_time = datetime.datetime.now()
    print(f"\n脚本完成于: {end_time.strftime('%Y-%m-%d_%H-%M-%S')}") # 中文打印
    print(f"总执行时间: {end_time - start_time}") # 中文打印


# --- 命令行参数解析 ---
if __name__ == "__main__":
    # 创建参数解析器
    # description 中文化
    parser = argparse.ArgumentParser(
        description='(无错误处理版本) 在 HDF5 点云样本上执行语义分割和可选聚类。'
    )

    # --- 定义输入/输出参数 ---
    # help 信息中文化
    parser.add_argument('--input_h5', type=str, default='./data/my_custom_dataset_h5_rgb/test_0.h5',
                        help='输入 HDF5 文件路径。')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='要处理的 HDF5 文件中的样本索引。')
    parser.add_argument('--checkpoint', type=str, default="checkpoints_seg_pyg_ptconv_rgb/best_model.pth",
                        help='预训练模型检查点 (.pth) 文件路径。')
    parser.add_argument('--output_dir', type=str, default='./segmentation_results_h5_no_try',
                        help='保存输出文件的目录 (如果使用 --save_results)。')

    # --- 定义数据处理参数 ---
    parser.add_argument('--num_points', type=int, default=40960,
                        help='期望的点数 (仅供参考, 代码会使用 HDF5 中的实际点数)。')
    parser.add_argument('--no_rgb', action='store_true',
                        help='不加载或使用 HDF5 中的 RGB 数据 (模型输入将是 3D)。')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='对点云 XYZ 坐标进行归一化 (中心化并缩放到单位球)。(默认: True)')

    # --- 定义模型相关参数 ---
    parser.add_argument('--num_classes', type=int, default=2,
                        help='模型训练时的语义类别数量。')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='(模型架构) 初始嵌入维度。')
    parser.add_argument('--k_neighbors', type=int, default=16,
                        help='(模型架构) k-NN 图中的 k 值。')
    parser.add_argument('--pt_hidden_dim', type=int, default=128,
                        help='(模型架构) Point Transformer 的隐藏层维度。')
    parser.add_argument('--pt_heads', type=int, default=4,
                        help='(模型架构) 注意力头数 (PyG PointTransformerConv 可能不直接使用)。')
    parser.add_argument('--num_transformer_layers', type=int, default=2,
                        help='(模型架构) Transformer 层的数量。')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='(模型架构) Dropout 比率。')

    # --- 定义聚类参数 ---
    parser.add_argument('--do_clustering', action='store_true', default=True,
                        help='在指定语义标签的点上启用 DBSCAN 聚类以寻找实例。')
    parser.add_argument('--screw_label_id', type=int, default=0,
                        help='要进行 DBSCAN 聚类的目标语义标签 ID。')
    parser.add_argument('--dbscan_eps', type=float, default=20,
                        help='DBSCAN 的 eps 参数 (邻域半径)。需要根据点云密度和归一化后的尺度调整。')
    parser.add_argument('--dbscan_min_samples', type=int, default=10,
                        help='DBSCAN 的 min_samples 参数 (形成核心点的最小邻居数)。需要根据预期实例大小调整。')

    # --- 定义控制参数 ---
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用 CUDA, 强制在 CPU 上运行。')
    parser.add_argument('--save_results', action='store_true',
                        help='保存处理后的点、语义标签、实例标签 (NPY, TXT) 以及每个实例的 PLY 文件 (如果启用聚类)。')

    # --- 定义可视化控制参数 ---
    parser.add_argument('--visualize_semantic', action='store_true', default=True,
                        help='在聚类/保存前显示语义分割结果 (按类别着色)。(默认: True)')
    parser.add_argument('--no_visualize', action='store_true', default=False,
                        help='禁用最终的 Open3D 可视化窗口 (通常显示实例分割结果)。')
    parser.add_argument('--marker_radius', type=float, default=0.01,
                        help='在最终可视化中用于标记实例质心的球体半径。')

    # 解析命令行传入的参数
    args = parser.parse_args()

    # --- 执行主逻辑 ---
    main(args)
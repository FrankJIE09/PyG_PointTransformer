# inference_orbbec_cluster.py
# 版本: 自动设备检测, 清理, 可选聚类区分实例, 可选可视化, 可选保存详细输出

import torch
import numpy as np
import argparse
import time
import os
import random
import datetime
import sys
import copy

# --- 导入必要的本地模块 ---
try:
    from camera.orbbec_camera import OrbbecCamera, get_serial_numbers, initialize_all_connected_cameras, close_connected_cameras
    from model import PyG_PointTransformerSegModel
except ImportError as e: print(f"FATAL Error importing local modules: {e}"); sys.exit(1)

# --- 导入 Open3D 和 Scikit-learn ---
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError: print("Warning: Open3D not found. Visualization and PLY saving disabled."); OPEN3D_AVAILABLE = False
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError: print("Warning: scikit-learn not found. Clustering disabled."); SKLEARN_AVAILABLE = False


# --- 预处理函数 (保持不变) ---
def preprocess_point_cloud(features_np, num_target_points, apply_normalization=False):
    # ... (与之前版本相同 - 处理 N x 3 或 N x 6 输入，返回 1 x N x 6 Tensor) ...
    if features_np is None or features_np.size == 0: return None
    if features_np.ndim != 2: return None
    current_channels = features_np.shape[1]
    if current_channels not in [3, 6]: return None

    current_num_points = features_np.shape[0]
    if current_num_points == num_target_points: processed_features = features_np
    elif current_num_points > num_target_points:
        indices = np.random.choice(current_num_points, num_target_points, replace=False)
        processed_features = features_np[indices, :]
    else:
        if current_num_points == 0: return None
        indices = np.random.choice(current_num_points, num_target_points, replace=True)
        processed_features = features_np[indices, :]

    if apply_normalization: print("Warning: Normalization placeholder.")

    final_channels = processed_features.shape[1]
    if final_channels == 3: # 如果输入只有 XYZ, 补充默认颜色
        default_colors = np.full((num_target_points, 3), 0.5, dtype=np.float32)
        processed_features = np.concatenate((processed_features, default_colors), axis=1)
    elif final_channels != 6: raise ValueError(f"Unexpected channels ({final_channels}) after processing.")

    features_tensor = torch.from_numpy(processed_features).float()
    features_tensor = features_tensor.unsqueeze(0)
    return features_tensor


# --- 可视化函数 (保持不变) ---
def visualize_segmentation(pcd_points_np, labels_np, num_classes,
                           label_map_colors, use_instance_labels=False,
                           instance_ids_np=None, window_name="Segmentation Result"):
    # ... (与之前版本相同) ...
    if not OPEN3D_AVAILABLE: print("Open3D not available."); return
    if pcd_points_np is None or labels_np is None or pcd_points_np.size == 0: return
    if pcd_points_np.shape[0] != labels_np.shape[0]: return
    if use_instance_labels and (instance_ids_np is None or pcd_points_np.shape[0] != instance_ids_np.shape[0]):
        print("Warning: Instance viz requested but IDs invalid. Fallback to semantic.")
        use_instance_labels = False

    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pcd_points_np[:, :3])

    if use_instance_labels and instance_ids_np is not None:
        print("Coloring by instance ID..."); unique_instance_ids = np.unique(instance_ids_np)
        max_instance_id = np.max(unique_instance_ids); instance_colors = np.random.rand(max_instance_id + 2, 3); instance_colors[0] = [0.3, 0.3, 0.3]
        point_colors = np.zeros_like(pcd_points_np);
        for i, inst_id in enumerate(instance_ids_np): point_colors[i] = instance_colors[inst_id + 1]
        pcd.colors = o3d.utility.Vector3dVector(point_colors); window_title = window_name + " (Instance Colors)"
    else:
        print("Coloring by semantic label..."); clamped_labels = np.clip(labels_np, 0, num_classes - 1)
        point_colors = label_map_colors[clamped_labels]; pcd.colors = o3d.utility.Vector3dVector(point_colors); window_title = window_name + " (Semantic Colors)"

    print("Displaying point cloud (Close Open3D window to continue)...")
    o3d.visualization.draw_geometries([pcd], window_name=window_title, width=800, height=600)
    print("Visualization window closed.")


# --- 主函数 ---
def main(args):
    print(f"Starting inference at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    # --- 依赖检查 ---
    if args.do_clustering and not SKLEARN_AVAILABLE: print("FATAL Error: Clustering requested but scikit-learn is not installed."); sys.exit(1)
    if (args.save_output or not args.no_visualize) and not OPEN3D_AVAILABLE: print("FATAL Error: PLY saving or Visualization requested but Open3D is not installed."); sys.exit(1)
    if args.visualize_instances and not args.do_clustering: print("Warning: --visualize_instances requires --do_clustering. Falling back to semantic viz."); args.visualize_instances = False

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"); print(f"Using device: {device}")

    # --- 加载模型 ---
    print(f"Loading SEMANTIC model checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint): raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
    print(f"Model structure initialized (Num Semantic Classes: {args.num_classes}, Expects 6D input).")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    else: model.load_state_dict(checkpoint)
    model.eval(); print("Model loaded successfully.")

    # --- 自动检测或使用指定设备 ---
    target_device_id = args.device_id
    # ... (自动检测逻辑不变) ...
    if target_device_id is None:
        print("No device_id provided, auto-detection..."); available_sns = get_serial_numbers()
        if len(available_sns) == 0: print("FATAL: No devices found."); return
        elif len(available_sns) == 1: target_device_id = available_sns[0]; print(f"Using SN: {target_device_id}")
        else: print(f"FATAL: Multiple devices found: {available_sns}. Use --device_id."); return
    else: print(f"Using specified device_id: {target_device_id}")

    # --- 初始化相机 ---
    print(f"Initializing Orbbec camera with SN/ID: {target_device_id}")
    camera = None
    try:
        cameras = initialize_all_connected_cameras([target_device_id])
        if cameras: camera = cameras[0]
        else: raise RuntimeError("Failed to initialize camera.")
        camera.start_stream(depth_stream=True, color_stream=True, use_alignment=True, enable_sync=True)
        print("Camera stream started.")

        # --- 预生成语义颜色映射表 ---
        np.random.seed(42)
        semantic_label_colors = np.random.rand(args.num_classes, 3)

        # --- 推理循环 ---
        print("\nEntering inference loop (Press Ctrl+C in terminal to quit)...")
        frame_count = 0
        start_loop_time = time.time()

        while True:
            loop_start_time = time.time()
            # 请求 XYZRGB 数据
            points_raw_np = camera.get_point_cloud(colored=True)

            if points_raw_np is None: time.sleep(0.01); continue

            # 预处理得到 6D 特征 Tensor
            input_features_tensor = preprocess_point_cloud(points_raw_np, args.num_points, apply_normalization=False)
            if input_features_tensor is None: continue

            # 同时保留处理后的 NumPy 坐标 (N, 3) 用于保存和聚类
            points_processed_np = input_features_tensor.squeeze(0).cpu().numpy()[:, :3]
            # 保留处理后的 NumPy 颜色 (N, 3) [0,1] 用于可能的保存 (可选)
            colors_processed_np = input_features_tensor.squeeze(0).cpu().numpy()[:, 3:]


            input_features_tensor = input_features_tensor.to(device)

            # 语义分割推理
            with torch.no_grad():
                logits = model(input_features_tensor)
            pred_semantic_labels_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy() # (N,)

            # --- 执行 DBSCAN 聚类 ---
            instance_labels_full = np.full_like(pred_semantic_labels_np, -1, dtype=np.int64) # 初始化为 -1
            num_instances_found = 0
            if args.do_clustering and SKLEARN_AVAILABLE:
                # print("\nPerforming DBSCAN clustering...") # 减少打印
                screw_label_id = args.screw_label_id
                screw_mask = (pred_semantic_labels_np == screw_label_id)
                screw_points_xyz = points_processed_np[screw_mask]

                if screw_points_xyz.shape[0] >= args.dbscan_min_samples:
                    db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
                    instance_labels_screw_points = db.fit_predict(screw_points_xyz)
                    instance_labels_full[screw_mask] = instance_labels_screw_points # 填充实例标签

                    unique_instances = np.unique(instance_labels_screw_points[instance_labels_screw_points != -1])
                    num_instances_found = len(unique_instances)
                    # num_noise_points = np.sum(instance_labels_screw_points == -1)
                    # print(f"Clustering found {num_instances_found} instances and {num_noise_points} noise points.")
                # else: # 点太少，instance_labels_full 保持 -1
                    # print(f"Not enough screw points ({screw_points_xyz.shape[0]}) for DBSCAN.")
                    pass


            # --- 保存输出文件 ---
            if args.save_output:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                output_dir = args.output_dir
                os.makedirs(output_dir, exist_ok=True)
                files_saved_msg = ["npy", "txt"] # 记录保存了哪些类型

                # 1. 保存 NPY
                np.save(os.path.join(output_dir, f"points_{timestamp}.npy"), points_processed_np)
                np.save(os.path.join(output_dir, f"semantic_labels_{timestamp}.npy"), pred_semantic_labels_np)
                # 只有在执行了聚类后才保存有意义的 instance_labels
                if args.do_clustering:
                     np.save(os.path.join(output_dir, f"instance_labels_{timestamp}.npy"), instance_labels_full)
                     files_saved_msg.append("instance_npy")


                # 2. 保存 TXT (XYZ, Semantic Label, Instance Label)
                data_to_save = np.hstack((
                    points_processed_np,
                    pred_semantic_labels_np.reshape(-1, 1),
                    instance_labels_full.reshape(-1, 1) # instance_labels_full 已包含 -1
                ))
                txt_filename = os.path.join(output_dir, f"segmented_full_{timestamp}.txt")
                np.savetxt(txt_filename, data_to_save, fmt='%.6f,%.6f,%.6f,%d,%d', delimiter=',')

                # 3. (修改) 按实例 ID 分别保存 PLY 文件 (可选着色)
                if args.do_clustering and OPEN3D_AVAILABLE:
                    unique_instances_plus_noise = np.unique(instance_labels_full)
                    num_ply_saved = 0
                    inst_colors = np.random.rand(len(unique_instances_plus_noise), 3) # 为每个出现的实例+噪声生成颜色
                    inst_colors[np.where(unique_instances_plus_noise == -1)] = [0.5, 0.5, 0.5] # 噪声设为灰色

                    for idx, inst_id in enumerate(unique_instances_plus_noise):
                        mask = (instance_labels_full == inst_id)
                        points_for_instance = points_processed_np[mask]
                        if points_for_instance.shape[0] == 0: continue

                        pcd_inst = o3d.geometry.PointCloud()
                        pcd_inst.points = o3d.utility.Vector3dVector(points_for_instance)
                        pcd_inst.paint_uniform_color(inst_colors[idx]) # 使用实例对应的统一颜色

                        ply_filename = os.path.join(output_dir, f"instance_{timestamp}_id_{inst_id}.ply")
                        try:
                            o3d.io.write_point_cloud(ply_filename, pcd_inst, write_ascii=False)
                            num_ply_saved += 1
                        except Exception as e_ply: print(f"\nError saving PLY for instance {inst_id}: {e_ply}")

                    if num_ply_saved > 0: files_saved_msg.append(f"{num_ply_saved}_instance_ply")

                print(f"Saved output ({', '.join(files_saved_msg)}) for frame {frame_count} with timestamp {timestamp}")
            # --- 结束保存 ---


            # --- 可视化 (可选) ---
            if not args.no_visualize:
                visualize_instances_now = args.visualize_instances and args.do_clustering and (instance_labels_full is not None)
                visualize_segmentation(
                    points_processed_np,           # 使用处理后的 XYZ
                    pred_semantic_labels_np,       # 语义标签
                    args.num_classes,
                    semantic_label_colors,       # 语义颜色图
                    use_instance_labels=visualize_instances_now,
                    instance_ids_np=instance_labels_full, # 实例标签
                    window_name=f"Frame {frame_count} Segmentation"
                )

            frame_count += 1
            loop_end_time = time.time()
            fps = 1.0 / (loop_end_time - loop_start_time) if (loop_end_time > loop_start_time) else 0
            print(f"\rProcessed frame {frame_count}. Raw points: {points_raw_np.shape[0]}, Instances found: {num_instances_found}, FPS: {fps:.2f}", end="")

            if not args.no_visualize: pass
            else: time.sleep(0.01)

    except KeyboardInterrupt: print("\nCtrl+C detected. Exiting loop.")
    except Exception as e: print(f"\nAn error occurred: {e}"); import traceback; traceback.print_exc()
    finally: # 清理
        print("\nCleaning up...");
        if camera: print("Stopping camera stream..."); camera.stop()
        # cv2 is not used for display anymore
        print("Inference loop finished.")


# --- 命令行参数解析 (保持不变) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time Point Cloud Instance Segmentation using Semantic Segmentation + Clustering')
    parser.add_argument('--checkpoint', type=str, default="checkpoints_seg_pyg_ptconv_rgb/best_model.pth",
                        help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--device_id', type=str, default=None, help='(Optional) Serial number of the Orbbec camera device.')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points the model expects')
    parser.add_argument('--num_classes', type=int,default=2, help='Number of SEMANTIC classes model was trained for (e.g., 2 for screw/background)')
    parser.add_argument('--k_neighbors', type=int, default=16, help='(Model Arch) k for k-NN graph')
    parser.add_argument('--embed_dim', type=int, default=64, help='(Model Arch) Initial embedding dimension')
    parser.add_argument('--pt_hidden_dim', type=int, default=128, help='(Model Arch) Hidden dimension for PointTransformerConv')
    parser.add_argument('--pt_heads', type=int, default=4, help='(Model Arch) Number of attention heads (if applicable)')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='(Model Arch) Number of PointTransformerConv layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='(Model Arch) Dropout rate')
    parser.add_argument('--do_clustering', action='store_true', help='Enable DBSCAN clustering to find instances.')
    parser.add_argument('--screw_label_id', type=int, default=1, help='The semantic label ID assigned to screws (default: 1)')
    parser.add_argument('--dbscan_eps', type=float, default=0.02, help='DBSCAN eps parameter (meters, default: 0.02)')
    parser.add_argument('--dbscan_min_samples', type=int, default=10, help='DBSCAN min_samples parameter (default: 10)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA inference (use CPU)')
    parser.add_argument('--no_visualize', action='store_true', help='Disable Open3D visualization window')
    parser.add_argument('--visualize_instances', action='store_true', help='Color visualization by instance ID (requires --do_clustering)')
    parser.add_argument('--save_output', action='store_true',default="True", help='Save points, labels (semantic/instance), and per-instance PLY files') # 更新 help
    parser.add_argument('--output_dir', type=str, default='./inference_output_clustered', help='Directory to save output files if --save_output is set') # 更新默认目录

    args = parser.parse_args()

    # --- 依赖检查 ---
    if args.do_clustering and not SKLEARN_AVAILABLE: print("FATAL Error: Clustering requested but scikit-learn not installed."); sys.exit(1)
    if (not args.no_visualize or args.save_output) and not OPEN3D_AVAILABLE: print("FATAL Error: Open3D is required for PLY saving or visualization but not installed."); sys.exit(1)
    if args.visualize_instances and not args.do_clustering: print("Warning: --visualize_instances requires --do_clustering. Falling back to semantic viz."); args.visualize_instances = False

    main(args)
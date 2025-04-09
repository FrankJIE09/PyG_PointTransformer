# inference_orbbec.py
# 版本: 自动设备检测, 基本清理, 打印示例标签, 可选按类别分别保存 PLY 输出

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
    # 确保路径正确
    from camera.orbbec_camera import OrbbecCamera, get_serial_numbers, initialize_all_connected_cameras, close_connected_cameras
    from model import PyG_PointTransformerSegModel
except ImportError as e:
    print(f"FATAL Error importing local modules (camera/orbbec_camera.py, model.py): {e}")
    sys.exit(1)

# --- 导入 Open3D ---
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: Open3D not found. Cannot save segmented PLY files or visualize.")
    OPEN3D_AVAILABLE = False


# --- 预处理函数 (保持不变) ---
def preprocess_point_cloud(points_np, num_target_points, apply_normalization=False):
    # ... (代码与上一个版本完全相同) ...
    if points_np is None or points_np.size == 0: return None
    if points_np.shape[1] > 3: points_np = points_np[:, :3]
    if points_np.shape[1] != 3: return None
    current_num_points = points_np.shape[0]
    if current_num_points == num_target_points: processed_points = points_np
    elif current_num_points > num_target_points:
        indices = np.random.choice(current_num_points, num_target_points, replace=False)
        processed_points = points_np[indices, :]
    else:
        if current_num_points == 0: return None
        indices = np.random.choice(current_num_points, num_target_points, replace=True)
        processed_points = points_np[indices, :]
    if apply_normalization: print("Warning: Normalization placeholder.")
    points_tensor = torch.from_numpy(processed_points).float()
    points_tensor = points_tensor.unsqueeze(0)
    return points_tensor


# --- 可视化函数 (保持不变，但注意它显示的是合并的点云) ---
def visualize_segmentation(pcd_points_np, pcd_labels_np, num_classes, window_name="Segmented Point Cloud"):
    # ... (代码与上一个版本完全相同) ...
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    if pcd_points_np is None or pcd_labels_np is None or pcd_points_np.size == 0 or pcd_labels_np.size == 0: return
    if pcd_points_np.shape[0] != pcd_labels_np.shape[0]: return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points_np)
    np.random.seed(42)
    colors = np.random.rand(num_classes, 3)
    clamped_labels = np.clip(pcd_labels_np, 0, num_classes - 1)
    point_colors = colors[clamped_labels]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    print("Displaying MERGED segmented point cloud (Close Open3D window to continue)...")
    o3d.visualization.draw_geometries([pcd], window_name=window_name, width=800, height=600)
    print("Visualization window closed.")


# --- 主函数 ---
def main(args):
    print(f"Starting inference at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- 加载模型 ---
    print(f"Loading model checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Error: Checkpoint file not found at {args.checkpoint}")
    model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
    print("Model structure initialized.")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    else: model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully.")

    # --- 自动检测或使用指定设备 ---
    target_device_id = args.device_id
    # ... (自动检测逻辑与上一版本相同) ...
    if target_device_id is None:
        print("No device_id provided, attempting auto-detection...")
        available_sns = get_serial_numbers()
        if len(available_sns) == 0: print("FATAL Error: No Orbbec devices found."); return
        elif len(available_sns) == 1:
            target_device_id = available_sns[0]; print(f"Found 1 device. Automatically using SN: {target_device_id}")
        else:
            print("FATAL Error: Found multiple Orbbec devices. Please specify one using --device_id argument:")
            for i, sn in enumerate(available_sns): print(f"  Device {i}: {sn}")
            return
    else: print(f"Using specified device_id: {target_device_id}")


    # --- 初始化相机 ---
    print(f"Initializing Orbbec camera with SN/ID: {target_device_id}")
    camera = None
    # 注意：实时可视化部分仍然显示合并的点云，或者可以禁用
    # vis = None # 如果需要非阻塞可视化，在这里初始化
    # pcd_vis = o3d.geometry.PointCloud()
    # is_geometry_added = False

    try:
        cameras = initialize_all_connected_cameras([target_device_id])
        if cameras: camera = cameras[0]
        else: raise RuntimeError("Failed to initialize camera.")
        camera.start_stream(depth_stream=True, color_stream=True, use_alignment=True, enable_sync=True)
        print("Camera stream started.")

        # --- 推理循环 ---
        print("\nEntering inference loop (Press Ctrl+C in terminal to quit)...")
        frame_count = 0
        start_loop_time = time.time()
        # --- 预先生成颜色映射表 ---
        np.random.seed(42) # 固定种子以获得一致的颜色
        label_to_color = np.random.rand(args.num_classes, 3)

        while True:
            loop_start_time = time.time()
            points_raw_np = camera.get_point_cloud(colored=False)

            if points_raw_np is None: time.sleep(0.01); continue
            input_tensor = preprocess_point_cloud(points_raw_np, args.num_points, apply_normalization=False)
            if input_tensor is None: continue

            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                logits = model(input_tensor)

            predictions = torch.argmax(logits, dim=2)
            pred_labels_np = predictions.squeeze(0).cpu().numpy() # (num_points,)
            points_for_processing = input_tensor.squeeze(0).cpu().numpy() # (num_points, 3)

            # --- 打印示例标签 (保持不变) ---
            # print("\n--- Point Labels Info (Sample Frame {}) ---".format(frame_count))
            # num_to_print = min(10, args.num_points)
            # for i in range(num_to_print):
            #     print(f"  Point {i}: Coord=[...], Predicted Label={pred_labels_np[i]}")
            # print("-----------------------------------------")

            # --- (修改) 根据参数保存输出 ---
            if args.save_output:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                output_dir = args.output_dir
                os.makedirs(output_dir, exist_ok=True)

                # 1. 保存完整的 .npy (坐标和标签)
                points_filename = os.path.join(output_dir, f"points_{timestamp}.npy")
                labels_filename = os.path.join(output_dir, f"labels_{timestamp}.npy")
                np.save(points_filename, points_for_processing)
                np.save(labels_filename, pred_labels_np)

                # 2. 保存完整的 .txt (坐标+标签)
                txt_filename = os.path.join(output_dir, f"segmented_all_{timestamp}.txt")
                data_to_save = np.hstack((points_for_processing, pred_labels_np.reshape(-1, 1)))
                np.savetxt(txt_filename, data_to_save, fmt='%.6f,%.6f,%.6f,%d', delimiter=',')

                # 3. (!!! 新增逻辑: 按类别分别保存 PLY 文件 !!!)
                if OPEN3D_AVAILABLE:
                    unique_labels = np.unique(pred_labels_np) # 获取当前帧实际出现的标签
                    num_saved_ply = 0
                    for label_id in unique_labels:
                        # a. 筛选出属于当前类别的点
                        mask = (pred_labels_np == label_id)
                        points_for_label = points_for_processing[mask]

                        if points_for_label.shape[0] == 0: # 如果没有点属于这个标签，跳过
                            continue

                        # b. 创建新的 Open3D 点云对象
                        pcd_label = o3d.geometry.PointCloud()
                        pcd_label.points = o3d.utility.Vector3dVector(points_for_label)

                        # c. (可选) 给这个类别的点云统一上色
                        label_color = label_to_color[label_id % args.num_classes] # 使用预生成的颜色
                        pcd_label.paint_uniform_color(label_color)

                        # d. 保存为单独的 PLY 文件
                        ply_filename = os.path.join(output_dir, f"segmented_{timestamp}_label_{label_id}.ply")
                        try:
                            o3d.io.write_point_cloud(ply_filename, pcd_label, write_ascii=False)
                            num_saved_ply += 1
                        except Exception as e_ply:
                            print(f"\nError saving PLY for label {label_id}: {e_ply}")

                    print(f"Saved NPY/TXT and {num_saved_ply} separate PLY files for frame {frame_count} with timestamp {timestamp}")
                else:
                     print(f"Saved NPY/TXT for frame {frame_count} with timestamp {timestamp}. PLY saving skipped (Open3D not available).")
            # --- 结束保存 ---

            # --- 可视化 (可选, 显示合并的点云) ---
            if not args.no_visualize:
                if OPEN3D_AVAILABLE:
                    visualize_segmentation(points_for_processing, pred_labels_np, args.num_classes)
                # else: # OPEN3D_AVAILABLE 为 False 时已在函数内处理
                #     pass # 无法可视化

            frame_count += 1
            loop_end_time = time.time()
            fps = 1.0 / (loop_end_time - loop_start_time) if (loop_end_time > loop_start_time) else 0
            print(f"\rProcessed frame {frame_count}. Input points: {points_raw_np.shape[0]}, Unique Labels: {np.unique(pred_labels_np).size}, FPS: {fps:.2f}", end="")

            if not args.no_visualize: pass
            else: time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting loop.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 清理 ---
        print("\nCleaning up...")
        if camera:
            print("Stopping camera stream...")
            camera.stop()
        # if vis: vis.destroy_window() # 如果使用非阻塞可视化
        end_time = time.time()
        # total_time = end_time - start_loop_time # 不准确
        print(f"Inference loop finished. Frames processed: {frame_count}")


# --- 命令行参数解析 (保持不变) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Real-time Point Cloud Segmentation Inference - Save by Label')

    parser.add_argument('--checkpoint', type=str, default="checkpoints_seg_pyg_ptconv/best_model.pth", help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--device_id', type=str, default=None, help='(Optional) Serial number of the Orbbec camera device.')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points the model expects')
    parser.add_argument('--num_classes', type=int, default=50, help='Number of segmentation classes model was trained for')
    parser.add_argument('--k_neighbors', type=int, default=16, help='(Model Arch) k for k-NN graph')
    parser.add_argument('--embed_dim', type=int, default=64, help='(Model Arch) Initial embedding dimension')
    parser.add_argument('--pt_hidden_dim', type=int, default=128, help='(Model Arch) Hidden dimension for PointTransformerConv')
    parser.add_argument('--pt_heads', type=int, default=4, help='(Model Arch) Number of attention heads')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='(Model Arch) Number of PointTransformerConv layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='(Model Arch) Dropout rate')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA inference (use CPU)')
    parser.add_argument('--no_visualize', action='store_true', help='Disable Open3D live visualization window')
    parser.add_argument('--save_output', action='store_true',default='True', help='Save points and predicted labels to files (NPY, TXT, and PLY per label)')
    parser.add_argument('--output_dir', type=str, default='./inference_output_bylabel', help='Directory to save output files if --save_output is set') # 改了默认目录名

    args = parser.parse_args()

    if (args.save_output or not args.no_visualize) and not OPEN3D_AVAILABLE:
        # 如果需要保存 PLY 或需要可视化，但 Open3D 不可用，则报错退出
        print("FATAL ERROR: Open3D is required for saving PLY output or visualization, but it's not installed.")
        sys.exit(1)
    elif args.no_visualize and not args.save_output:
         print("Running inference without visualization and without saving output.")
    # else: # Open3D is available or not needed
    #     pass

    main(args)
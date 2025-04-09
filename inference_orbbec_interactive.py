# inference_orbbec_interactive.py
# 版本: 自动设备检测, 基本清理, 周期性打印标签 (移除了键盘回调)

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
except ImportError as e:
    print(f"FATAL Error importing local modules (orbbec_camera.py, model.py): {e}")
    sys.exit(1)

# --- 导入 Open3D ---
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("FATAL Error: Open3D not found. This script requires Open3D for visualization.")
    sys.exit(1)

# --- 全局变量 (不再需要为回调函数共享，但在主循环中仍使用) ---
current_points_np = None
current_labels_np = None
current_num_classes = 50

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

# --- (移除) 点选回调函数 ---
# def pick_points_callback(vis): ... # 不再需要

# --- 主函数 (修改可视化和信息打印逻辑) ---
def main(args):
    global current_points_np, current_labels_np, current_num_classes # 声明使用全局变量

    print(f"Starting inference at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")
    current_num_classes = args.num_classes

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
    if target_device_id is None:
        print("No device_id provided, attempting auto-detection...")
        available_sns = get_serial_numbers()
        if len(available_sns) == 0: print("FATAL Error: No Orbbec devices found."); return
        elif len(available_sns) == 1:
            target_device_id = available_sns[0]
            print(f"Found 1 device. Automatically using SN: {target_device_id}")
        else:
            print("FATAL Error: Found multiple Orbbec devices. Please specify one using --device_id argument:")
            for i, sn in enumerate(available_sns): print(f"  Device {i}: {sn}")
            return
    else:
        print(f"Using specified device_id: {target_device_id}")

    # --- 初始化相机和 Visualizer ---
    print(f"Initializing Orbbec camera with SN/ID: {target_device_id}")
    camera = None
    vis = None
    pcd = o3d.geometry.PointCloud()
    is_geometry_added = False

    try:
        cameras = initialize_all_connected_cameras([target_device_id])
        if cameras: camera = cameras[0]
        else: raise RuntimeError("Failed to initialize camera.")
        camera.start_stream(depth_stream=True, color_stream=True, use_alignment=True, enable_sync=True)
        print("Camera stream started.")

        print("\nInitializing Open3D visualizer...")
        vis = o3d.visualization.Visualizer() # 使用基础 Visualizer
        vis.create_window(window_name='Real-time Segmentation', width=800, height=600)
        print("Open3D window created.")
        # --- 不再注册键盘回调 ---
        # vis.register_key_callback(...)

        print("\nEntering inference loop (Press Ctrl+C in terminal to quit)...")
        frame_count = 0
        start_loop_time = time.time()
        print_interval = 100 # 每 100 帧打印一次信息

        while True:
            loop_start_time = time.time()
            points_raw_np = camera.get_point_cloud(colored=False)

            if points_raw_np is None:
                time.sleep(0.01)
                # 即使没有新点云也要处理窗口事件
                if not vis.poll_events(): break
                vis.update_renderer()
                continue

            input_tensor = preprocess_point_cloud(points_raw_np, args.num_points, apply_normalization=False)
            if input_tensor is None:
                if not vis.poll_events(): break
                vis.update_renderer()
                continue

            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                logits = model(input_tensor)

            predictions = torch.argmax(logits, dim=2)
            pred_labels_np = predictions.squeeze(0).cpu().numpy()
            points_for_viz = input_tensor.squeeze(0).cpu().numpy()

            # 更新全局变量（虽然不再有回调函数直接使用，但保留无妨）
            current_points_np = points_for_viz
            current_labels_np = pred_labels_np

            # --- (新增) 周期性打印标签信息 ---
            if frame_count % print_interval == 0:
                print("\n--- Point Labels Info (Sample Frame {}) ---".format(frame_count))
                num_to_print = min(5, args.num_points) # 打印前 5 个
                for i in range(num_to_print):
                    point_coord = points_for_viz[i]
                    point_label = pred_labels_np[i]
                    print(f"  Point {i}: Coord=[{point_coord[0]:.3f} {point_coord[1]:.3f} {point_coord[2]:.3f}], Predicted Label={point_label}")
                print("-----------------------------------------")
            # --- 结束周期性打印 ---

            # --- (新增) 根据参数保存输出 (保持不变) ---
            if args.save_output:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                output_dir = args.output_dir
                os.makedirs(output_dir, exist_ok=True)
                points_filename = os.path.join(output_dir, f"points_{timestamp}.npy")
                labels_filename = os.path.join(output_dir, f"labels_{timestamp}.npy")
                np.save(points_filename, points_for_viz)
                np.save(labels_filename, pred_labels_np)
                txt_filename = os.path.join(output_dir, f"segmented_{timestamp}.txt")
                data_to_save = np.hstack((points_for_viz, pred_labels_np.reshape(-1, 1)))
                np.savetxt(txt_filename, data_to_save, fmt='%.6f,%.6f,%.6f,%d', delimiter=',')
                # print(f"Saved output files for frame {frame_count}...") # 减少打印

            # --- 更新 Open3D 可视化 ---
            pcd.points = o3d.utility.Vector3dVector(points_for_viz)
            np.random.seed(42)
            colors = np.random.rand(args.num_classes, 3)
            clamped_labels = np.clip(pred_labels_np, 0, args.num_classes - 1)
            point_colors = colors[clamped_labels]
            pcd.colors = o3d.utility.Vector3dVector(point_colors)

            if not is_geometry_added:
                vis.add_geometry(pcd)
                is_geometry_added = True
            else:
                vis.update_geometry(pcd)

            # 处理窗口事件
            if not vis.poll_events():
                break
            vis.update_renderer()

            frame_count += 1
            loop_end_time = time.time()
            fps = 1.0 / (loop_end_time - loop_start_time) if (loop_end_time > loop_start_time) else 0
            print(f"\rProcessed frame {frame_count}. Input points: {points_raw_np.shape[0]}, FPS: {fps:.2f}", end="")
            # time.sleep(0.01) # 在非阻塞可视化下可能不再需要

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
        if vis:
            print("Destroying Open3D window...")
            vis.destroy_window()
        end_time = time.time()
        # total_time = end_time - start_loop_time
        print(f"Inference loop finished. Frames processed: {frame_count}")


# --- 命令行参数解析 (移除 --no_visualize) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Real-time Point Cloud Segmentation with Orbbec Camera and PyG Model (Periodic Print)')

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
    # --- 新增保存参数 ---
    parser.add_argument('--save_output', action='store_true', help='Save points and predicted labels to files')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='Directory to save output files if --save_output is set')

    args = parser.parse_args()

    if not OPEN3D_AVAILABLE:
         print("FATAL ERROR: Open3D is required for this script.")
    else:
         main(args)
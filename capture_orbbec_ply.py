# capture_orbbec_ply.py
# 版本: 自动设备检测, 基本清理, 按 R 键保存带 RGB 的 PLY (可选体素下采样)

import time
import datetime
import os
import argparse
import sys

import cv2 # 用于图像显示和按键检测
import numpy as np

# 假设 orbbec_camera.py 在 camera 子目录下
try:
    from camera.orbbec_camera import OrbbecCamera, get_serial_numbers, initialize_all_connected_cameras, close_connected_cameras
except ImportError as e:
    print(f"FATAL Error importing from camera.orbbec_camera: {e}")
    sys.exit(1)

# 需要 Open3D 来保存 PLY 文件和执行下采样
try:
    import open3d as o3d
except ImportError:
    print("FATAL Error: Open3D not found. Required for saving PLY and voxel downsampling.")
    sys.exit(1)


def main(args):
    """主函数，执行捕获和保存逻辑"""
    print(f"Starting PLY capture script at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    # --- 1. 创建输出目录 ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"PLY files (XYZRGB) will be saved to: {os.path.abspath(output_dir)}")
    if args.voxel_size > 0:
        print(f"Voxel downsampling enabled with voxel size: {args.voxel_size:.4f} meters")
    else:
        print("Voxel downsampling disabled.")

    # --- 2. 自动检测或使用指定设备 ---
    target_device_id = args.device_id
    # ... (自动检测逻辑不变) ...
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

    # --- 3. 初始化相机 ---
    print(f"Initializing Orbbec camera with SN/ID: {target_device_id}")
    camera = None
    try:
        cameras = initialize_all_connected_cameras([target_device_id])
        if cameras: camera = cameras[0]
        else: raise RuntimeError("Failed to initialize camera.")
        camera.start_stream(depth_stream=True, color_stream=True, use_alignment=True, enable_sync=True)
        print("Camera stream started.")

        # --- 4. 捕获和保存循环 ---
        print("\n--- Live Preview ---")
        print(" Press [R] in the preview window to save the current point cloud as PLY.")
        print(" Press [Q] or [ESC] in the preview window to quit.")
        print("---------------------")

        frame_count = 0
        saved_count = 0
        window_name = f"Orbbec RGB Preview (SN: {target_device_id})"

        while True:
            color_image, _, depth_frame = camera.get_frames()

            # 显示彩色预览图
            display_image = None
            if color_image is not None:
                display_image = color_image
            else:
                 # ... (创建黑色占位符图像的代码不变) ...
                if camera.depth_profile:
                     h, w = camera.depth_profile.get_height(), camera.depth_profile.get_width()
                     display_image = np.zeros((h, w, 3), dtype=np.uint8); cv2.putText(display_image, 'No Color Image', (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                else: print("Error: Cannot get stream dimensions."); break

            if display_image is not None:
                 cv2.imshow(window_name, display_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27: # ESC
                print("\nQuit key pressed. Exiting...")
                break

            # --- 按 R 保存 XYZRGB 点云 ---
            if key == ord('r') or key == ord('R'):
                print(f"\n[R] key pressed. Capturing and saving XYZRGB point cloud...")

                points_np_xyzrgb = camera.get_point_cloud(colored=True)

                if points_np_xyzrgb is not None and points_np_xyzrgb.size > 0:
                    original_point_count = points_np_xyzrgb.shape[0]
                    pcd = o3d.geometry.PointCloud() # 创建空的 Open3D 点云对象

                    # 检查返回的形状并填充 pcd 对象
                    if points_np_xyzrgb.ndim == 2:
                        if points_np_xyzrgb.shape[1] == 6: # XYZRGB
                            points_xyz = points_np_xyzrgb[:, :3]
                            points_rgb = points_np_xyzrgb[:, 3:6]
                            points_rgb_normalized = np.clip(points_rgb / 255.0, 0.0, 1.0)
                            pcd.points = o3d.utility.Vector3dVector(points_xyz)
                            pcd.colors = o3d.utility.Vector3dVector(points_rgb_normalized)
                            save_suffix = "_xyzrgb"
                        elif points_np_xyzrgb.shape[1] == 3: # XYZ only
                            print("Warning: Got only XYZ points despite requesting color.")
                            pcd.points = o3d.utility.Vector3dVector(points_np_xyzrgb)
                            save_suffix = "_xyz"
                        else:
                            print(f"Error: Unexpected number of columns: {points_np_xyzrgb.shape[1]}")
                            continue # 跳过这个保存操作
                    else:
                        print(f"Error: Unexpected point cloud dimensions: {points_np_xyzrgb.shape}")
                        continue

                    # --- (!!! 新增: 体素下采样 !!!) ---
                    pcd_to_save = pcd # 默认保存原始（或处理后）的点云
                    downsampled_point_count = original_point_count # 默认为原始点数

                    if args.voxel_size > 0:
                        print(f"  Applying voxel downsampling with voxel size: {args.voxel_size:.4f}...")
                        pcd_downsampled = pcd.voxel_down_sample(voxel_size=args.voxel_size)
                        downsampled_point_count = len(pcd_downsampled.points)
                        if downsampled_point_count == 0:
                            print("  Warning: Voxel downsampling resulted in 0 points. Skipping save.")
                            continue # 不保存空点云
                        else:
                            pcd_to_save = pcd_downsampled # 更新为保存下采样后的点云
                            print(f"  Downsampled from {original_point_count} to {downsampled_point_count} points.")
                    # --- 结束下采样 ---

                    # 创建文件名
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    filename = f"{args.prefix}_{timestamp}{save_suffix}.ply" # 根据是否有颜色添加后缀
                    filepath = os.path.join(output_dir, filename)

                    # 保存为 PLY 文件
                    try:
                        o3d.io.write_point_cloud(filepath, pcd_to_save, write_ascii=False, compressed=False) # 明确不压缩
                        print(f"Successfully saved: {filepath} ({downsampled_point_count} points)")
                        saved_count += 1
                    except Exception as e_save:
                        print(f"Error saving PLY file {filepath}: {e_save}")

                    # (可选) 保存对应的彩色图像
                    if args.save_color and color_image is not None:
                        img_filename = f"{args.prefix}_{timestamp}.png"
                        img_filepath = os.path.join(output_dir, img_filename)
                        try: cv2.imwrite(img_filepath, color_image); print(f"Successfully saved color image: {img_filepath}")
                        except Exception as e_img_save: print(f"Error saving color image {img_filepath}: {e_img_save}")

                else:
                    print("Failed to get valid point cloud for saving.")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 清理 ---
        print("\nCleaning up...")
        if camera:
            print("Stopping camera stream...")
            camera.stop()
        cv2.destroyAllWindows()
        print(f"Capture session finished. Total PLY files saved: {saved_count}")


# --- 命令行参数解析 (新增 --voxel_size) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture Point Clouds from Orbbec Camera and save as PLY on key press, with optional voxel downsampling.')

    parser.add_argument('--output_dir', type=str, default='./capture_output_xyzrgb',
                        help='Directory to save the captured PLY files (default: ./capture_output_xyzrgb)')
    parser.add_argument('--device_id', type=str, default=None,
                        help='(Optional) Serial number of the Orbbec camera device.')
    parser.add_argument('--prefix', type=str, default='capture',
                        help='Prefix for the saved PLY filenames (default: capture)')
    parser.add_argument('--save_color', action='store_true',
                        help='Also save the corresponding color image as PNG when saving PLY.')
    parser.add_argument('--no_rgb_preview', action='store_true',
                        help='Disable the live RGB preview window (disables key detection).')
    # --- 新增下采样参数 ---
    parser.add_argument('--voxel_size', type=float, default=0,
                        help='Voxel size for downsampling in meters. Set to 0 or negative to disable downsampling (default: 0.0)')

    args = parser.parse_args()

    if args.no_rgb_preview:
        print("\nWarning: --no_rgb_preview is set. Key detection (R to save, Q/ESC to quit) requires the OpenCV window.")
        # Let it run, but keys won't work. User must use Ctrl+C.
        pass

    # if not OPEN3D_AVAILABLE:
    #      print("FATAL Error: Open3D is required for saving PLY files and downsampling.")
    #      sys.exit(1)

    main(args)
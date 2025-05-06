# -*- coding: utf-8 -*-
import time
import numpy as np
import cv2
import open3d as o3d
import open3d.core as o3c
import open3d.t.pipelines.slam as slam # <--- 引入 slam 模块

# Import from your Orbbec camera interface file
try:
    from camera.orbbec_camera_test import (
        OrbbecCamera,
        get_serial_numbers,
        initialize_all_connected_cameras,
        close_connected_cameras,
        MIN_DEPTH,
        MAX_DEPTH
    )
    print("Successfully imported from camera/orbbec_camera_test.py")
except ImportError as e:
    print(f"Error importing from camera/orbbec_camera_test.py: {e}")
    print("Please ensure orbbec_camera_test.py is in the 'camera' subdirectory or Python path.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit()

# --- Configuration ---
VOXEL_SIZE = 0.03
DEPTH_SCALE = 1000.0
DEPTH_MAX_M = MAX_DEPTH / DEPTH_SCALE # 最大深度（米）
DEPTH_TRUNC_M = VOXEL_SIZE * 5.0
BLOCK_RESOLUTION = 16  # 每个 Voxel Block 的分辨率 (通常是 16)
INITIAL_BLOCK_COUNT = 10000 # 初始分配的 Voxel Block 数量 (可调整)

# --- Odometry Options ---
ODOMETRY_METHOD = o3d.pipelines.odometry.OdometryOption()
# -----------------------

VIS_UPDATE_RATE = 0.2

# --- GPU/CPU Device Setup ---
try:
    if o3c.cuda.is_available():
        device = o3c.Device("CUDA:0")
        print("GPU CUDA support found. Using CUDA device.")
    else:
        device = o3c.Device("CPU:0")
        print("GPU CUDA support not found or Open3D not built with CUDA. Using CPU device.")
except Exception as e:
    print(f"Error checking CUDA availability: {e}. Defaulting to CPU.")
    device = o3c.Device("CPU:0")
# ---------------------------

def main():
    print("Starting 3D Reconstruction Example (using VoxelBlockGrid)...") # <--- 修改标题

    # --- 1. Initialize Camera ---
    # ... (相机初始化代码不变) ...
    available_sns = get_serial_numbers()
    if not available_sns: print("Error: No Orbbec devices found."); return
    print(f"Found devices: {available_sns}")
    orbbec_cam = initialize_all_connected_cameras([available_sns[0]])
    if not orbbec_cam: print(f"Error: Failed to initialize camera {available_sns[0]}."); return
    camera = orbbec_cam[0]
    print(f"Successfully initialized camera: {camera.get_serial_number()}")

    vis = None

    try:
        # --- 2. Start Camera Stream ---
        # ... (启动流的代码不变) ...
        print("Starting camera stream (Color, Depth)...")
        camera.start_stream(
            depth_stream=True, color_stream=True, enable_accel=False,
            enable_gyro=False, use_alignment=True, enable_sync=True
        )
        time.sleep(1)
        if not camera.is_streaming(): print("Error: Camera stream failed to start."); return
        print("Camera stream started successfully.")


        # --- 3. Get Camera Intrinsics ---
        # ... (获取内参并创建 legacy 和 tensor 内参的代码不变) ...
        print("Getting camera intrinsics...")
        cam_params_ob = camera.param
        if cam_params_ob is None: print("Error: Failed to get camera parameters."); return
        width = camera.color_profile.get_width(); height = camera.color_profile.get_height()
        fx = cam_params_ob.rgb_intrinsic.fx; fy = cam_params_ob.rgb_intrinsic.fy
        cx = cam_params_ob.rgb_intrinsic.cx; cy = cam_params_ob.rgb_intrinsic.cy
        if width <= 0 or height <= 0 or fx <= 0 or fy <= 0 or cx <= 0 or cy <= 0:
             print("Error: Invalid camera intrinsic values obtained."); return
        intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        intrinsics_t = o3c.Tensor(intrinsics_o3d.intrinsic_matrix, o3c.Dtype.Float64, device)
        print("Open3D legacy and tensor intrinsics created.")

        # --- 4. Initialize Odometry Variables ---
        # ... (Odometry 变量初始化不变) ...
        current_absolute_pose = np.identity(4)
        previous_rgbd_o3d = None

        # --- 5. Initialize Reconstruction (VoxelBlockGrid) --- <--- 修改点
        print(f"Initializing VoxelBlockGrid on device {device.get_type()}:{device.get_id()}...")
        # 定义 VoxelBlockGrid 的属性，通常与 TSDF 类似
        voxel_grid = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.uint16, o3c.uint8),
            # *** 新增 attr_channels 参数 ***
            attr_channels=([1, 1, 3]),  # 通道数: tsdf=1, weight=1, color=3
            voxel_size=VOXEL_SIZE,
            block_resolution=BLOCK_RESOLUTION,
            block_count=INITIAL_BLOCK_COUNT,
            device=device
        )
        print("VoxelBlockGrid initialized.")

        # --- 6. Setup Visualization ---
        # ... (可视化器设置不变) ...
        vis = o3d.visualization.Visualizer()
        vis.create_window("Live 3D Reconstruction (VoxelBlockGrid)", width=1280, height=720) # <--- 修改标题
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.002, origin=[0, 0, 0])
        vis.add_geometry(origin_frame)
        print("**DEBUG: Added coordinate frame to visualizer.**")
        point_cloud_vis = o3d.geometry.PointCloud()
        vis_geom_added = False
        last_vis_update_time = time.time()
        print("Open3D visualizer created.")

        print("\nStarting main loop. Point camera and move SLOWLY. Press 'q' in OpenCV window to exit.")

        # --- 7. Main Processing Loop ---
        frame_count = 0
        while True:
            loop_start_time = time.time()

            # --- 7.1 Get Frames ---
            # ... (获取帧代码不变) ...
            color_image, depth_data_mm, _, accel_data, gyro_data = camera.get_frames()
            if color_image is None or depth_data_mm is None:
                if vis and not vis.poll_events(): break
                if vis: vis.update_renderer()
                cv2.waitKey(1); continue

            # --- 7.2 Prepare Data ---
            # ... (准备 legacy RGBDImage 和 Tensor 数据的代码不变) ...
            color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(depth_data_mm)
            rgbd_current_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=MAX_DEPTH + 1.0, convert_rgb_to_intensity=False
            )
            color_t = o3c.Tensor.from_numpy(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)).to(device)
            depth_t = o3c.Tensor.from_numpy(depth_data_mm).to(device, o3c.Dtype.UInt16)

            # --- 7.3 Estimate Pose (Basic RGBD Odometry - CPU) ---
            # ... (Odometry 代码不变) ...
            relative_pose = np.identity(4)
            odometry_success = False
            if previous_rgbd_o3d is not None:
                [odometry_success, relative_pose, info_matrix] = o3d.pipelines.odometry.compute_rgbd_odometry(
                    previous_rgbd_o3d, rgbd_current_o3d, intrinsics_o3d, np.identity(4),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), ODOMETRY_METHOD
                )
                if not odometry_success or np.isnan(relative_pose).any() or np.isinf(relative_pose).any():
                    if frame_count > 1:
                       if not odometry_success: print(f"**DEBUG Frame {frame_count}: Odometry failed.**")
                       else: print(f"**DEBUG Frame {frame_count}: Odometry returned invalid pose.**")
                    relative_pose = np.identity(4)
                    odometry_success = False
            current_absolute_pose = current_absolute_pose @ relative_pose


            # --- 7.4 Integrate Frame into VoxelBlockGrid using slam.integrate --- <--- 修改点
            integration_attempted = False
            integration_success = False
            if frame_count == 0 or odometry_success:
                integration_attempted = True
                try:
                   current_pose_inv = np.linalg.inv(current_absolute_pose)
                   extrinsic_t = o3c.Tensor(current_pose_inv.astype(np.float32), o3c.Dtype.Float32, device)

                   # 使用 slam.integrate 函数融合数据到 voxel_grid (in-place modification)
                   slam.integrate(
                       depth_t,           # Depth tensor (UInt16)
                       color_t,           # Color tensor (UInt8)
                       voxel_grid,        # The VoxelBlockGrid object to integrate into
                       intrinsics_t,      # Intrinsics tensor (Float64)
                       extrinsic_t,       # Extrinsic tensor (Float32)
                       depth_scale=DEPTH_SCALE, # mm -> m (Float32)
                       depth_max=DEPTH_MAX_M*1000,   # Max depth in meters (Float32)
                       sdf_trunc=DEPTH_TRUNC_M  # SDF truncation distance in meters
                       # block_coords = None # Optional: Specify which blocks to integrate
                   )
                   integration_success = True
                except np.linalg.LinAlgError as lae:
                    print(f"**ERROR Frame {frame_count}: Linear algebra error during pose inversion: {lae}. Skipping integration.**")
                except AttributeError as ae:
                     # 捕获可能的 slam.integrate 缺失错误
                     if "'slam' has no attribute 'integrate'" in str(ae) or \
                        "'module' object has no attribute 'integrate'" in str(ae):
                         print("**FATAL ERROR: 'slam.integrate' function not found in Open3D installation.**")
                         print("Your Open3D installation might be incomplete regarding SLAM pipelines.")
                         print("Please reinstall Open3D with full features or revert to CPU implementation.")
                         break # 退出循环
                     else:
                         print(f"**ERROR Frame {frame_count}: AttributeError during integration: {ae}**")
                except Exception as e:
                    print(f"**ERROR Frame {frame_count}: Error during VoxelBlockGrid integration: {e}**")


            # --- 7.5 Update Visualization --- <--- 修改点
            current_time = time.time()
            if current_time - last_vis_update_time > VIS_UPDATE_RATE:
                try:
                    # 从 VoxelBlockGrid 提取点云，需要权重阈值
                    pcd_t = voxel_grid.extract_point_cloud(weight_threshold=3.0) # 尝试阈值 3.0 (可调整)
                    new_cloud = pcd_t.to_legacy() if pcd_t is not None else None
                except Exception as e:
                    print(f"**ERROR Frame {frame_count}: Error extracting point cloud from VoxelBlockGrid: {e}")
                    new_cloud = None # 确保 new_cloud 是 None

                num_points = 0
                if new_cloud:
                    num_points = len(new_cloud.points)

                # ... (更新点云和检查消息的代码不变, 除了上面提取部分) ...
                if num_points > 0:
                    point_cloud_vis.points = new_cloud.points
                    point_cloud_vis.colors = new_cloud.colors
                    if not vis_geom_added:
                        vis.add_geometry(point_cloud_vis)
                        vis_geom_added = True
                        print(f"****** Frame {frame_count}: Point cloud geometry ADDED to visualizer! ({num_points} points) ******")
                    else:
                        vis.update_geometry(point_cloud_vis)
                elif integration_attempted and integration_success and frame_count > 0:
                     print(f"**WARNING Frame {frame_count}: Integration succeeded but extracted cloud is EMPTY ({num_points} points). Try adjusting weight_threshold?**")


                last_vis_update_time = current_time

            if not vis.poll_events():
                print("Visualizer window closed by user.")
                break
            vis.update_renderer()

            # ... (显示 2D 图像和深度图的代码不变) ...
            cv2.imshow("Orbbec Color Feed", color_image)
            depth_display = cv2.normalize(depth_data_mm, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imshow("Input Depth (mm, visualized)", depth_display)


            # --- 7.6 Prepare for Next Iteration ---
            previous_rgbd_o3d = rgbd_current_o3d
            frame_count += 1

            # --- 7.7 Check for Exit Key ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exit key 'q' pressed.")
                break

        # --- End of Main Loop ---
        print("Exiting main loop.")

        # --- 8. Final Output (Optional) --- <--- 修改点
        if frame_count > 0:
            try:
                # 从 VoxelBlockGrid 提取最终点云
                final_pcd_t = voxel_grid.extract_point_cloud(weight_threshold=3.0) # 使用相同的阈值
                final_pcd = final_pcd_t.to_legacy() if final_pcd_t is not None else None

                if final_pcd and len(final_pcd.points) > 0:
                    output_filename = "final_reconstruction_vbg.ply" # <--- 修改文件名
                    print(f"Saving final point cloud to {output_filename}...")
                    o3d.io.write_point_cloud(output_filename, final_pcd)
                    print(f"Point cloud with {len(final_pcd.points)} points saved.")
                else:
                     print("Final point cloud extraction yielded no points. Not saving.")
            except Exception as e:
                print(f"Error during final save: {e}")

    except Exception as e:
        print(f"\nAn critical error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        # ... (清理代码不变) ...
        print("\nCleaning up resources...")
        if vis: vis.destroy_window(); print("Visualizer window destroyed.")
        close_connected_cameras([camera])
        cv2.destroyAllWindows()
        print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()
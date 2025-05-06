# -*- coding: utf-8 -*-
import time
import numpy as np
import cv2
import open3d as o3d # Import Open3D

# Import from your Orbbec camera interface file
try:
    from camera.orbbec_camera_test import (
        OrbbecCamera,
        get_serial_numbers,
        initialize_all_connected_cameras,
        close_connected_cameras,
        MIN_DEPTH, # Import depth limits if needed by reconstruction
        MAX_DEPTH
    )
    print("Successfully imported from orbbec_camera_test.py")
except ImportError as e:
    print(f"Error importing from orbbec_camera_test.py: {e}")
    print("Please ensure orbbec_camera_test.py is in the same directory or Python path.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit()

# --- Configuration ---
VOXEL_SIZE = 0.03
DEPTH_SCALE = 1.0
DEPTH_TRUNC = VOXEL_SIZE * 5.0

# --- Odometry Options ---
ODOMETRY_METHOD = o3d.pipelines.odometry.OdometryOption()
# ODOMETRY_METHOD.max_depth_diff = 0.05 # 稍微放宽限制，可以尝试调整
# print(f"Using OdometryOption with default settings (or modified max_depth_diff={ODOMETRY_METHOD.max_depth_diff}).")
# -----------------------

VIS_UPDATE_RATE = 0.2


def main():
    print("Starting 3D Reconstruction Example...")

    # --- 1. Initialize Camera ---
    available_sns = get_serial_numbers()
    if not available_sns:
        print("Error: No Orbbec devices found.")
        return
    print(f"Found devices: {available_sns}")
    orbbec_cam = initialize_all_connected_cameras([available_sns[0]])
    if not orbbec_cam:
        print(f"Error: Failed to initialize camera {available_sns[0]}.")
        return
    camera = orbbec_cam[0]
    print(f"Successfully initialized camera: {camera.get_serial_number()}")

    vis = None

    try:
        # --- 2. Start Camera Stream ---
        print("Starting camera stream (Color, Depth)...")
        camera.start_stream(
            depth_stream=True,
            color_stream=True,
            enable_accel=False,
            enable_gyro=False,
            use_alignment=True, # CRITICAL
            enable_sync=True
        )
        time.sleep(1)

        if not camera.is_streaming():
            print("Error: Camera stream failed to start.")
            return
        print("Camera stream started successfully.")

        # --- 3. Get Camera Intrinsics ---
        print("Getting camera intrinsics...")
        cam_params_ob = camera.param
        if cam_params_ob is None:
            print("Error: Failed to get camera parameters.")
            return

        width = camera.color_profile.get_width()
        height = camera.color_profile.get_height()
        fx = cam_params_ob.rgb_intrinsic.fx
        fy = cam_params_ob.rgb_intrinsic.fy
        cx = cam_params_ob.rgb_intrinsic.cx
        cy = cam_params_ob.rgb_intrinsic.cy

        if width <= 0 or height <= 0 or fx <= 0 or fy <= 0 or cx <= 0 or cy <= 0:
             print("Error: Invalid camera intrinsic values obtained.")
             print(f"W: {width}, H: {height}, fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
             return

        intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        print("Open3D camera intrinsics created:")
        # print(intrinsics_o3d.intrinsic_matrix) # Optional

        # --- 4. Initialize Odometry Variables ---
        current_absolute_pose = np.identity(4)
        previous_rgbd_o3d = None

        # --- 5. Initialize Reconstruction (TSDF Volume) ---
        print(f"Initializing TSDF Volume with Voxel Size: {VOXEL_SIZE}m")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=VOXEL_SIZE,
            sdf_trunc=DEPTH_TRUNC,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        print("TSDF Volume initialized.")

        # --- 6. Setup Visualization ---
        vis = o3d.visualization.Visualizer()
        vis.create_window("Live 3D Reconstruction", width=1280, height=720)

        # **DEBUG:** Add coordinate frame to test visualizer
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        vis.add_geometry(origin_frame)
        print("**DEBUG: Added coordinate frame to visualizer.**") # 检查坐标系是否添加成功

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
            color_image, depth_data_mm, _, accel_data, gyro_data = camera.get_frames()

            if color_image is None or depth_data_mm is None:
                # **DEBUG:** 打印跳过信息
                # print(f"DEBUG Frame {frame_count}: Skipping (Color: {color_image is not None}, Depth: {depth_data_mm is not None})")
                if vis and not vis.poll_events(): break
                if vis: vis.update_renderer()
                cv2.waitKey(1)
                continue
            else:
                # **DEBUG:** 检查深度数据的有效范围
                valid_depth_count = np.count_nonzero((depth_data_mm > MIN_DEPTH) & (depth_data_mm < MAX_DEPTH))
                total_pixels = depth_data_mm.size
                valid_ratio = valid_depth_count / total_pixels if total_pixels > 0 else 0
                # print(f"DEBUG Frame {frame_count}: Valid depth ratio = {valid_ratio:.2%}") # 可以取消注释来查看

            # --- 7.2 Prepare Data for Open3D ---
            color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(depth_data_mm)

            rgbd_current_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=DEPTH_SCALE,
                depth_trunc=(MAX_DEPTH / DEPTH_SCALE), # 使用导入的 MAX_DEPTH
                convert_rgb_to_intensity=False
            )
            # **DEBUG:** 检查转换后的 RGBD 图像是否包含数据
            # if not rgbd_current_o3d.has_depth() or not rgbd_current_o3d.has_color():
            #      print(f"**WARNING Frame {frame_count}: create_from_color_and_depth resulted in missing data!**")
                 # continue # 如果这里有问题，可以选择跳过这一帧


            # --- 7.3 Estimate Pose (Basic RGBD Odometry) ---
            relative_pose = np.identity(4)
            odometry_success = False

            if previous_rgbd_o3d is not None:
                # **DEBUG:** 打印开始计算 Odometry
                # print(f"DEBUG Frame {frame_count}: Computing Odometry...")
                [odometry_success, relative_pose, info_matrix] = o3d.pipelines.odometry.compute_rgbd_odometry(
                    previous_rgbd_o3d,
                    rgbd_current_o3d,
                    intrinsics_o3d,
                    np.identity(4),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    ODOMETRY_METHOD
                )

                # **DEBUG:** 打印 Odometry 结果
                if not odometry_success or np.isnan(relative_pose).any() or np.isinf(relative_pose).any():
                    if frame_count > 1:
                       if not odometry_success: print(f"**DEBUG Frame {frame_count}: Odometry failed.**")
                       else: print(f"**DEBUG Frame {frame_count}: Odometry returned invalid pose.**")
                    relative_pose = np.identity(4)
                    odometry_success = False
                # else: # 成功时打印
                #     print(f"DEBUG Frame {frame_count}: Odometry OK.")

            # Update absolute pose
            current_absolute_pose = current_absolute_pose @ relative_pose

            # --- 7.4 Integrate Frame into TSDF Volume ---
            integration_attempted = False
            integration_success = False
            if frame_count == 0 or odometry_success: # 只有第一帧或 Odometry 成功时才积分
                integration_attempted = True
                # print(f"**DEBUG Frame {frame_count}: Attempting integration.**")
                try:
                   # **DEBUG:** 检查积分前的位姿是否有效
                   if np.isnan(current_absolute_pose).any() or np.isinf(current_absolute_pose).any():
                       print(f"**ERROR Frame {frame_count}: Invalid absolute pose before integration! Skipping integration.**")
                   else:
                       volume.integrate(
                           rgbd_current_o3d,
                           intrinsics_o3d,
                           np.linalg.inv(current_absolute_pose) # Extrinsic = Pose^-1
                       )
                       integration_success = True # 假设 integrate 不报错即成功
                       # print(f"**DEBUG Frame {frame_count}: Integration call successful.**")
                except np.linalg.LinAlgError as lae:
                    print(f"**ERROR Frame {frame_count}: Linear algebra error during pose inversion: {lae}. Skipping integration.**")
                except Exception as e:
                    print(f"**ERROR Frame {frame_count}: Error during TSDF integration: {e}**")


            # --- 7.5 Update Visualization ---
            current_time = time.time()
            if current_time - last_vis_update_time > VIS_UPDATE_RATE:
                new_cloud = volume.extract_point_cloud()

                num_points = 0
                if new_cloud:
                    num_points = len(new_cloud.points)

                # **DEBUG:** 打印提取和积分状态
                # print(f"**DEBUG Frame {frame_count}: Update Vis - Extracted {num_points} points. Int Attempt: {integration_attempted}, Int Success: {integration_success}, Odom Success: {odometry_success}")

                if num_points > 0:
                    point_cloud_vis.points = new_cloud.points
                    point_cloud_vis.colors = new_cloud.colors
                    if not vis_geom_added:
                        vis.add_geometry(point_cloud_vis)
                        vis_geom_added = True
                        # **DEBUG:** 打印添加成功的消息
                        print(f"****** Frame {frame_count}: Point cloud geometry ADDED to visualizer! ({num_points} points) ******")
                    else:
                        vis.update_geometry(point_cloud_vis)
                        # print(f"DEBUG Frame {frame_count}: Point cloud geometry updated ({num_points} points).") # 可以取消注释查看更新
                elif integration_attempted and integration_success and frame_count > 0:
                     # 如果尝试并成功积分了，但提取点云为空，打印警告
                     print(f"**WARNING Frame {frame_count}: Integration succeeded but extracted cloud is EMPTY ({num_points} points).**")

                last_vis_update_time = current_time

            if not vis.poll_events():
                print("Visualizer window closed by user.")
                break
            vis.update_renderer()

            # Show 2D color image
            cv2.imshow("Orbbec Color Feed", color_image)

            # **DEBUG:** Show input depth map
            depth_display = cv2.normalize(depth_data_mm, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imshow("Input Depth (mm, visualized)", depth_display)

            # --- 7.6 Prepare for Next Iteration ---
            previous_rgbd_o3d = rgbd_current_o3d # 存储当前帧供下次使用
            frame_count += 1

            # --- 7.7 Check for Exit Key ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exit key 'q' pressed.")
                break

        # --- End of Main Loop ---
        print("Exiting main loop.")

        # --- 8. Final Output (Optional) ---
        if frame_count > 0:
            try:
                final_pcd = volume.extract_point_cloud()
                if final_pcd and len(final_pcd.points) > 0:
                    output_filename = "final_reconstruction.ply"
                    print(f"Saving final point cloud to {output_filename}...")
                    o3d.io.write_point_cloud(output_filename, final_pcd)
                    print(f"Point cloud with {len(final_pcd.points)} points saved.")
                else:
                     print("Final point cloud extraction yielded no points. Not saving.") # 增加无点提示
            except Exception as e:
                print(f"Error during final save: {e}")

    except Exception as e:
        print(f"\nAn critical error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        print("\nCleaning up resources...")
        if vis:
            vis.destroy_window()
            print("Visualizer window destroyed.")
        close_connected_cameras([camera]) # Use the function from your file
        cv2.destroyAllWindows() # Close OpenCV windows including the depth one
        print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()
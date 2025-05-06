# -*- coding: utf-8 -*-
import time                     # 导入时间库
import cv2                      # 导入 OpenCV 库
import numpy as np              # 导入 NumPy 库
import open3d as o3d            # 导入 Open3D 库
import sys                      # 导入系统库 (用于处理导入路径等)

# --- 假设 ---
# 1. 你的原始 'orbbec_camera.py' 文件与此脚本在同一目录，或在其 Python 搜索路径中。
# 2. 'orbbec_camera.py' 内部所需的 'camera/utils.py' 文件也存在。
# 3. 你已安装好 pyorbbecsdk, numpy, opencv-python, pyyaml, 和 open3d。

try:
    # 从你的 orbbec_camera.py 文件中导入所需的类和变量
    # OrbbecCamera: 你已经定义好的相机控制类
    # get_serial_numbers: 用于查找设备序列号的函数
    # MIN_DEPTH, MAX_DEPTH: 在 orbbec_camera.py 中定义的深度过滤常量
    from camera.orbbec_camera import OrbbecCamera, get_serial_numbers, MIN_DEPTH, MAX_DEPTH
except ImportError as e:
    print(f"错误：无法从 'orbbec_camera.py' 导入所需内容: {e}")
    print("请确保 'orbbec_camera.py' 文件在此脚本的同一目录或 Python 路径中，")
    print("并且它内部依赖的 'camera/utils.py' 文件也存在。")
    sys.exit(1) # 导入失败则退出


def realtime_point_cloud_viewer(serial_number=None, use_colored_point_cloud=True, fix_depth_filter=True):
    """
    (这个函数调用你写的 OrbbecCamera 类)
    初始化一个 Orbbec 相机, 启动数据流, 并使用 Open3D 连续捕获和实时可视化点云,
    允许用户手动移动相机进行扫描。

    Args:
        serial_number (str, optional): 要使用的特定相机的序列号。
                                       如果为 None, 则使用找到的第一个相机。默认为 None。
        use_colored_point_cloud (bool): 如果为 True, 尝试生成并显示 XYZRGB 彩色点云。
                                        需要相机支持彩色流并且启用了对齐。默认为 True。
        fix_depth_filter (bool): 如果为 True, 尝试在此函数内修正性地应用深度过滤器
                                 (使用 orbbec_camera.py 中的 MIN_DEPTH, MAX_DEPTH)。
                                 这假设 Z 坐标是米(m)，而常量是毫米(mm)。默认为 True。
                                 如果你的 get_point_cloud 内部已经正确处理了过滤, 可以设为 False。
    """
    camera = None   # 相机对象变量
    vis = None      # Open3D 可视化器对象变量

    try:
        # --- 1. 初始化相机 (使用你文件中的 OrbbecCamera 类) ---
        if serial_number is None:
            print("未提供序列号, 正在搜索相机...")
            sns = get_serial_numbers() # 调用你文件中的函数
            if not sns:
                print("错误: 未找到任何 Orbbec 相机。")
                return
            serial_number = sns[0]
            print(f"将使用找到的第一个相机: {serial_number}")
        else:
            print(f"尝试使用指定相机: {serial_number}")

        try:
            # 创建 OrbbecCamera 类的实例
            camera = OrbbecCamera(serial_number)
            print(f"成功初始化相机: {camera.get_device_name()} ({serial_number})")
        except RuntimeError as e:
            print(f"初始化相机 {serial_number} 出错: {e}")
            return

        # --- 2. 启动数据流 (调用你类中的 start_stream 方法) ---
        # 如果需要彩色点云，必须启用彩色流和对齐
        enable_color = use_colored_point_cloud
        try:
            print("正在启动相机流 (深度, 彩色, 对齐)...")
            # 调用相机对象的 start_stream 方法
            camera.start_stream(depth_stream=True,         # 启用深度流
                                color_stream=enable_color, # 根据需要启用彩色流
                                use_alignment=enable_color,# 如果启用彩色，通常也需启用对齐
                                enable_sync=True)          # 启用帧同步
        except RuntimeError as e:
            print(f"启动流出错: {e}")
            return
        except Exception as e:
            print(f"启动流时发生意外错误: {e}")
            return

        # --- 3. 设置 Open3D 可视化器 ---
        print("正在设置 Open3D 可视化器...")
        vis = o3d.visualization.Visualizer() # 创建可视化器对象
        vis.create_window(window_name=f"Orbbec 实时点云 - {serial_number}", width=1280, height=720) # 创建窗口

        # 创建一个空的点云几何对象 (只需一次)
        pcd = o3d.geometry.PointCloud()
        first_frame = True # 标记是否是第一帧数据

        print("\n可视化器已启动。请移动相机进行扫描。")
        print("关闭 Open3D 窗口即可停止。")

        # --- 4. 实时处理循环 ---
        while True:
            # --- 获取点云数据 (调用你类中的 get_point_cloud 方法) ---
            # 注意: get_point_cloud 方法内部会使用它自己的 PointCloudFilter
            points = camera.get_point_cloud(colored=use_colored_point_cloud)

            # 检查是否成功获取到点云数据
            if points is not None and points.shape[0] > 0:
                # --- 可选: 在此应用修正后的深度过滤器 ---
                # 因为 OrbbecCamera.get_point_cloud 内部的过滤器可能存在单位问题 (米 vs 毫米)
                # 如果 fix_depth_filter 为 True，我们在这里根据米单位重新过滤一次
                if fix_depth_filter:
                    z_coordinates = points[:, 2] # 假设 Z 轴是第3列 (索引2), 单位是米
                    min_depth_m = MIN_DEPTH / 1000.0 # 将 orbbec_camera.py 定义的 MIN_DEPTH (mm) 转为米
                    max_depth_m = MAX_DEPTH / 1000.0 # 将 orbbec_camera.py 定义的 MAX_DEPTH (mm) 转为米
                    depth_mask = (z_coordinates >= min_depth_m) & (z_coordinates <= max_depth_m)
                    points = points[depth_mask] # 应用过滤器

                # 检查过滤后是否还有点
                if points.shape[0] == 0:
                     if not first_frame: # 如果不是第一帧 (避免清除空对象)
                        pcd.clear() # 清除几何体内容
                        vis.update_geometry(pcd) # 更新显示为空白
                     continue # 跳过此帧，处理下一帧

                # --- 更新点云几何对象的点和颜色 ---
                # 更新点坐标 (XYZ)
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])

                # 如果是彩色点云且数据包含6列
                if use_colored_point_cloud and points.shape[1] == 6:
                    # 假设 Orbbec SDK 输出的 RGB 颜色在 0-255 范围, 需归一化到 0-1
                    colors = points[:, 3:6] / 255.0
                    colors = np.clip(colors, 0.0, 1.0) # 确保颜色值在 [0, 1] 区间
                    pcd.colors = o3d.utility.Vector3dVector(colors) # 更新颜色
                else:
                    # 如果非彩色或颜色数据缺失，给点云一个默认的灰色
                    pcd.paint_uniform_color([0.7, 0.7, 0.7])

                # --- 更新 Open3D 显示 ---
                if first_frame:
                    # 如果是第一帧有效数据, 将几何对象添加到场景中
                    vis.add_geometry(pcd)
                    first_frame = False # 不再是第一帧
                else:
                    # 对于后续帧, 只需更新已存在的几何对象
                    vis.update_geometry(pcd)

            else:
                # 如果未收到点云数据 (例如帧被丢弃或过滤后为空)
                # 可选: 清除显示内容
                if not first_frame: # 避免清除空对象
                    pcd.clear()
                    vis.update_geometry(pcd)
                # print("此帧未接收到点云数据。") # 频繁打印会很吵

            # --- 处理窗口事件 ---
            # 这对于保持窗口响应 (如旋转、缩放、关闭) 至关重要
            if not vis.poll_events():
                print("可视化窗口已被用户关闭。")
                break # 如果窗口关闭，则退出循环

            vis.update_renderer() # 请求重新绘制场景

            # 可以根据需要添加微小的延时, 但 poll_events 通常已足够
            # time.sleep(0.01)

    except KeyboardInterrupt: # 捕获 Ctrl+C 中断
        print("\n检测到 Ctrl+C。正在停止。")
    except Exception as e: # 捕获其他意外错误
        print(f"\n主循环中发生意外错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误信息
    finally:
        # --- 5. 清理资源 ---
        print("正在进行清理...")
        if vis:
            # 销毁 Open3D 窗口
            vis.destroy_window()
            print("可视化窗口已销毁。")
        if camera and camera.stream:
            # 如果相机对象存在且流是开启的，则停止流
            try:
                camera.stop() # 调用你类中的 stop 方法
                print("相机流已停止。")
            except Exception as e:
                print(f"停止相机流时出错: {e}")
        # 关闭可能由代码其他部分打开的 OpenCV 窗口
        cv2.destroyAllWindows()
        print("清理完成。")

# --- 如何运行这个函数的示例 ---
if __name__ == "__main__":
    # === 选择一种方式运行 ===

    # 方式 1: 使用找到的第一个相机, 生成彩色点云, 并应用修正的深度过滤
    # realtime_point_cloud_viewer()

    # 方式 2: 指定你的相机序列号 (替换 "YOUR_CAMERA_SERIAL_NUMBER")
    # my_serial_number = "YOUR_CAMERA_SERIAL_NUMBER"
    # realtime_point_cloud_viewer(serial_number=my_serial_number)

    # 方式 3: 使用第一个相机, 但只生成 XYZ 点云 (非彩色)
    realtime_point_cloud_viewer(use_colored_point_cloud=False)

    # 方式 4: 使用第一个相机, 彩色点云, 但 *不* 在此函数中应用修正的深度过滤
    # (假设你已经在 OrbbecCamera.get_point_cloud 内部修复了过滤逻辑)
    # realtime_point_cloud_viewer(use_colored_point_cloud=True, fix_depth_filter=False)
import cv2
import numpy as np
import open3d as o3d
# import rospy # 取消注释以启用ROS功能
# from std_srvs.srv import Trigger, TriggerResponse # 示例ROS服务
# from your_package.srv import EstimatePose, EstimatePoseResponse # 您的自定义ROS服务消息 (如果需要)
import subprocess
import os
import sys
import time
import tempfile
import re  # 用于解析输出
import yaml

# 假设 orbbec_camera.py 在 'camera' 子目录中
try:
    from camera.orbbec_camera import OrbbecCamera, get_serial_numbers, initialize_all_connected_cameras, \
        close_connected_cameras
except ImportError as e:
    print(f"错误: 无法导入 OrbbecCamera 模块: {e}")
    print("请确保 'camera' 文件夹及 'orbbec_camera.py' 文件与此脚本在同一目录下，或者已正确安装。")
    sys.exit(1)

# --- 配置 ---
PART1_SCRIPT_NAME = "_estimate_pose_part1_preprocessing_pca.py"
PART2_SCRIPT_NAME = "_estimate_pose_part2_icp_estimation.py"
CONFIG_YAML_FILE = "pose_estimation_config.yaml"  # 假设与脚本在同一目录或可访问路径

# 此协调器脚本的输出目录
ORCHESTRATOR_BASE_OUTPUT_DIR = "./orchestrator_run_output"


# ROS全局变量 (示例)
# last_pose_estimation_result = None

# def estimate_pose_service_callback(req):
#     global last_pose_estimation_result
#     # ... 在此触发处理流程或返回 last_pose_estimation_result ...
#     # 当前版本主要通过 'r' 键触发
#     pass

def find_script_path(script_name):
    """尝试在多个常见位置查找脚本或配置文件。"""
    # 1. 尝试相对于当前工作目录
    path_in_cwd = os.path.join(os.getcwd(), script_name)
    if os.path.exists(path_in_cwd) and os.path.isfile(path_in_cwd):
        return os.path.abspath(path_in_cwd)

    # 2. 尝试相对于此脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_in_script_dir = os.path.join(script_dir, script_name)
    if os.path.exists(path_in_script_dir) and os.path.isfile(path_in_script_dir):
        return os.path.abspath(path_in_script_dir)

    # 3. 如果是脚本名，尝试直接使用（假设在系统PATH中）
    #    对于配置文件，通常不应依赖PATH。
    if script_name.endswith(".py"):  # 粗略判断是否为脚本
        print(f"警告: 脚本 {script_name} 在常规位置未找到。假设它在系统PATH中或可以被python直接执行。")
        return script_name  # 返回原始名称，让subprocess尝试查找

    print(f"错误: 文件 {script_name} 未在预期的位置找到。")
    return None


def run_command_and_parse_output(command_list, parse_regex_dict, timeout_seconds=300):
    """
    运行一个命令并从其stdout中解析多个正则表达式模式。
    parse_regex_dict: {"key_name": "regex_pattern_with_one_capture_group", ...}
    返回: 解析值的字典 {"key_name": "captured_value", ...} 或在失败时返回 None
    """
    print(f"执行命令: {' '.join(command_list)}")
    results = {}
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   encoding='utf-8')
        stdout, stderr = process.communicate(timeout=timeout_seconds)

        if process.returncode != 0:
            print(f"错误: 执行命令失败: {' '.join(command_list)}")
            print(f"返回码: {process.returncode}")
            print("Stdout:")
            print(stdout)
            print("Stderr:")
            print(stderr)
            return None

        print("命令成功执行。Stdout:")
        print(stdout)
        if stderr:
            print("Stderr (可能包含警告或次要错误):")
            print(stderr)

        all_parsed = True
        for key, pattern in parse_regex_dict.items():
            match = re.search(pattern, stdout)
            if match and match.group(1):
                results[key] = match.group(1).strip()
                print(f"  已解析 '{key}': {results[key]}")
            else:
                print(f"  警告: 无法从stdout中使用正则表达式 '{pattern}' 解析 '{key}'。")
                results[key] = None
                all_parsed = False  # 标记至少有一个解析失败

        # 如果任何关键解析失败，可能也指示一个问题
        # if not all_parsed:
        # print("  警告:并非所有预期的输出都被成功解析。")

        return results

    except subprocess.TimeoutExpired:
        print(f"错误: 命令执行超时: {' '.join(command_list)}")
        if 'process' in locals() and process:
            process.kill()
            stdout_timeout, stderr_timeout = process.communicate()
            print("超时时的Stdout:")
            print(stdout_timeout)
            print("超时时的Stderr:")
            print(stderr_timeout)
        return None
    except FileNotFoundError:
        print(f"错误: 无法找到命令/脚本: {command_list[0]}. 请检查路径和Python环境。")
        return None
    except Exception as e:
        print(f"错误: 运行命令 {' '.join(command_list)} 时发生异常: {e}")
        return None


def main_orchestrator():
    # global last_pose_estimation_result # ROS 相关

    # --- 初始化 ROS (如果需要) ---
    # rospy.init_node('pose_estimation_orchestrator', anonymous=True)
    # pose_service = rospy.Service('/estimate_pose_service', EstimatePose, estimate_pose_service_callback) # 示例
    print("提示: ROS相关功能在此版本中为占位符。")

    # --- 相机初始化 ---
    print("正在初始化奥比中光相机...")
    camera_instance = None
    try:
        available_sns = get_serial_numbers()
        if not available_sns:
            print("错误: 未找到任何奥比中光设备。请检查连接。正在退出。")
            return
        print(f"找到的设备序列号: {available_sns}。将使用第一个: {available_sns[0]}")
        camera_instance = OrbbecCamera(available_sns[0])  # 使用列表中的第一个相机
        camera_instance.start_stream(depth_stream=True, color_stream=True, use_alignment=True, enable_sync=True)
    except Exception as e:
        print(f"错误: 初始化奥比中光相机失败: {e}")
        print("请确保奥比中光相机已连接，并且SDK已正确安装配置。")
        print("您可能需要在脚本中指定一个特定的设备序列号。")
        if camera_instance:  # 尝试停止以防部分启动
            try:
                camera_instance.stop()
            except:
                pass
        return

    print("相机初始化成功。在OpenCV窗口中按 'r' 键捕捉点云并开始处理流程。按 'q' 键退出。")

    os.makedirs(ORCHESTRATOR_BASE_OUTPUT_DIR, exist_ok=True)

    part1_script_path = find_script_path(PART1_SCRIPT_NAME)
    part2_script_path = find_script_path(PART2_SCRIPT_NAME)
    config_yaml_path = find_script_path(CONFIG_YAML_FILE)

    if not all([part1_script_path, part2_script_path, config_yaml_path]):
        print("错误: 一个或多个必需的脚本/配置文件未找到。请检查路径。正在退出。")
        if camera_instance: camera_instance.stop()
        return

    try:
        while True:
            color_image, _, _ = camera_instance.get_frames()

            display_image = color_image
            if display_image is None:
                # print("未能从相机获取彩色图像。") # 频繁打印可能会影响可读性
                display_image = np.zeros((480, 640, 3), dtype=np.uint8)  # 显示黑色图像以保持窗口
                cv2.putText(display_image, "No Camera Feed", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Orbbec Color Feed (Press 'r' to process, 'q' to quit)", display_image)

            key = cv2.waitKey(10) & 0xFF  # 等待10ms
            if key == ord('q'):
                print("检测到 'q' 键，正在退出...")
                break
            elif key == ord('r'):
                print("\n检测到 'r' 键。正在捕获点云并启动处理流程...")

                points_xyzrgb = camera_instance.get_point_cloud(colored=True)

                if points_xyzrgb is None or points_xyzrgb.shape[0] == 0:
                    print("错误: 从相机获取有效点云失败。请重试。")
                    continue

                print(f"成功捕获点云，点数: {points_xyzrgb.shape[0]}，维度: {points_xyzrgb.shape[1]}")

                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(points_xyzrgb[:, :3])
                if points_xyzrgb.shape[1] == 6:  # 检查是否有颜色数据
                    colors = points_xyzrgb[:, 3:6]
                    # 检查颜色值范围，Orbbec SDK的get_point_cloud通常输出0-255范围的RGB
                    if np.max(colors) > 1.0:
                        colors = colors / 255.0
                    colors = np.clip(colors, 0.0, 1.0)  # 确保在[0,1]范围内
                    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
                else:
                    print("点云不包含颜色信息，将使用默认灰色。")
                    pcd_o3d.paint_uniform_color([0.5, 0.5, 0.5])

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                current_run_output_dir = os.path.join(ORCHESTRATOR_BASE_OUTPUT_DIR, f"run_{timestamp}")
                os.makedirs(current_run_output_dir, exist_ok=True)

                temp_ply_file_path = os.path.join(current_run_output_dir, f"captured_scene_{timestamp}.ply")
                try:
                    o3d.io.write_point_cloud(temp_ply_file_path, pcd_o3d)
                    print(f"捕获的点云已保存至: {temp_ply_file_path}")
                except Exception as e_save_pcd:
                    print(f"错误: 保存捕获的点云失败: {e_save_pcd}")
                    continue

                # --- 运行 Part 1 ---
                print("\n--- 正在运行 Part 1: 预处理和 PCA ---")
                part1_processing_output_dir = os.path.join(current_run_output_dir, "part1_intermediate_data")
                os.makedirs(part1_processing_output_dir, exist_ok=True)

                cmd_part1 = [
                    sys.executable, part1_script_path,
                    "--input_point_cloud_file", temp_ply_file_path,
                    "--config_file", config_yaml_path,
                    "--output_dir", part1_processing_output_dir,  # Part1会在此目录下创建其特定的子目录
                    # "--no_cuda" # 如果需要强制CPU运行Part1，取消此行注释
                    # "--visualize_semantic_segmentation", "False", # 控制Part1的内部可视化
                ]

                parse_dict_part1 = {"intermediate_dir": r"Intermediate data will be saved to: (.*)"}
                part1_results = run_command_and_parse_output(cmd_part1, parse_dict_part1)

                if not part1_results or not part1_results.get("intermediate_dir"):
                    print("错误: Part 1 执行失败或未能解析出中间数据目录。跳过 Part 2。")
                    continue

                intermediate_dir_from_part1 = part1_results["intermediate_dir"]
                if not os.path.isdir(intermediate_dir_from_part1):
                    print(f"错误: Part 1报告的中间目录不存在: {intermediate_dir_from_part1}")
                    # 尝试基于输入文件名和Part1的输出逻辑来预测路径 (作为备选方案)
                    base_input_name_for_part1 = os.path.splitext(os.path.basename(temp_ply_file_path))[0]
                    predicted_intermediate_dir = os.path.join(part1_processing_output_dir,
                                                              f"intermediate_data_{base_input_name_for_part1}")
                    print(f"尝试使用预测的中间目录路径: {predicted_intermediate_dir}")
                    if os.path.isdir(predicted_intermediate_dir):
                        intermediate_dir_from_part1 = predicted_intermediate_dir
                        print("已切换到使用预测的中间目录。")
                    else:
                        print("预测的中间目录也不存在。无法继续执行 Part 2。")
                        continue

                # --- 运行 Part 2 ---
                print("\n--- 正在运行 Part 2: ICP 位姿估计 ---")
                part2_final_results_dir = os.path.join(current_run_output_dir, "part2_final_output")
                os.makedirs(part2_final_results_dir, exist_ok=True)

                cmd_part2 = [
                    sys.executable, part2_script_path,
                    "--intermediate_dir", intermediate_dir_from_part1,
                    "--config_file", config_yaml_path,
                    "--output_dir_part2", part2_final_results_dir,  # Part 2 将在此保存NPZ文件
                    "--visualize_pose",  # 启用Part2的ICP对齐可视化
                    "--visualize_pose_in_scene",  # 启用Part2的场景中姿态可视化
                    "--save_results",  # 确保Part2保存NPZ结果
                ]

                parse_dict_part2 = {"npz_file": r"Saved PyTorch ICP poses to (.*\.npz)"}
                part2_results = run_command_and_parse_output(cmd_part2, parse_dict_part2)

                if not part2_results or not part2_results.get("npz_file"):
                    print("错误: Part 2 执行失败或未能解析出NPZ文件路径。")
                    continue

                npz_file_path = part2_results["npz_file"]
                if not os.path.exists(npz_file_path):
                    print(f"错误: Part 2报告的NPZ文件不存在: {npz_file_path}")
                    # 备选方案: 在part2_final_results_dir中搜索最新的.npz文件
                    npz_files_in_output = [f for f in os.listdir(part2_final_results_dir) if f.endswith('.npz')]
                    if npz_files_in_output:
                        npz_files_in_output.sort(
                            key=lambda f_name: os.path.getmtime(os.path.join(part2_final_results_dir, f_name)))
                        npz_file_path = os.path.join(part2_final_results_dir, npz_files_in_output[-1])
                        print(f"警告: 使用找到的最新NPZ文件作为备选: {npz_file_path}")
                        if not os.path.exists(npz_file_path):  # 再次检查
                            print("备选NPZ文件路径也无效。无法加载位姿。")
                            continue
                    else:
                        print(f"在目录 {part2_final_results_dir} 中未找到任何NPZ文件。无法加载位姿。")
                        continue

                # --- 处理结果 ---
                print(f"\n--- 正在处理来自 {npz_file_path} 的结果 ---")
                try:
                    estimated_poses_data = np.load(npz_file_path)
                    print(f"成功从 {npz_file_path} 加载位姿数据。")
                    if not estimated_poses_data.files:
                        print("警告: NPZ文件中未找到任何位姿 (无实例)。")

                    for instance_key in estimated_poses_data.files:
                        pose_matrix = estimated_poses_data[instance_key]
                        print(f"实例 '{instance_key}' 的估计位姿矩阵:\n{pose_matrix}")
                        # last_pose_estimation_result = {"instance": instance_key, "pose": pose_matrix} # 用于ROS Service

                        # 在此处执行ROS Service调用或TF广播
                        print(f"ROS集成占位符: 现在将为实例 {instance_key} 传输TF。")
                        # 示例 (需要取消注释并配置rospy):
                        # if rospy.is_shutdown(): continue # 检查ROS是否仍在运行
                        # try:
                        #     # 假设您有一个名为 /update_tf_service 的服务
                        #     # 并且有一个名为 UpdateTF.srv 的服务定义: string instance_id, float64[16] transform_matrix -> bool success, string message
                        #     rospy.wait_for_service('/update_tf_service', timeout=1.0)
                        #     update_tf_proxy = rospy.ServiceProxy('/update_tf_service', UpdateTF) # 替换为您的服务类型
                        #     response = update_tf_proxy(instance_id=instance_key, transform_matrix=pose_matrix.flatten().tolist())
                        #     print(f"ROS Service 调用成功: {response.message if response else 'No response'}")
                        # except rospy.ServiceException as e_ros_srv:
                        #     print(f"ROS Service 调用失败: {e_ros_srv}")
                        # except rospy.ROSException as e_ros_timeout: # 例如 wait_for_service 超时
                        #     print(f"ROS错误 (可能是服务不可用): {e_ros_timeout}")
                        # except NameError: # 如果UpdateTF未定义 (因为rospy未完全集成)
                        #     print("ROS服务类型 UpdateTF 未定义。跳过服务调用。")

                except Exception as e_load_npz:
                    print(f"错误: 加载或处理NPZ文件 {npz_file_path} 失败: {e_load_npz}")

                print("\n当前捕获和处理流程已完成。")
                print("请在OpenCV窗口按 'r' 键进行下一次处理，或按 'q' 键退出。")

    except KeyboardInterrupt:
        print("\n检测到Ctrl+C，正在退出...")
    finally:
        if camera_instance:
            print("正在停止相机...")
            camera_instance.stop()
        cv2.destroyAllWindows()
        print("协调器程序已结束。")


if __name__ == "__main__":
    main_orchestrator()
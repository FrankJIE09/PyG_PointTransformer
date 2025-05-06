# realtime_orbbec_scan.py
# 依赖: pip install pyorbbecsdk open3d opencv-python numpy
import numpy as np
import open3d as o3d
import cv2
import signal
from camera.orbbec_camera import OrbbecCamera, \
    get_serial_numbers  # 同目录导入  :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}

VOXEL_SIZE_INTEGRATE = 0.003  # m   全局体素下采样
VOXEL_SIZE_ICP = 0.005  # m   ICP 用低分辨率更快
MAX_CORR_DIST = 0.02  # m   ICP 最大对应点距离


def init_camera() -> OrbbecCamera:
    sns = get_serial_numbers()
    if not sns:
        raise RuntimeError("未检测到任何奥比中光相机")
    cam = OrbbecCamera(sns[0])
    # 开启深度+彩色+对齐，同步帧时间
    cam.start_stream(depth_stream=True, color_stream=True,
                     use_alignment=True, enable_sync=True)
    return cam


def make_pcd_from_numpy(points_np: np.ndarray) -> o3d.geometry.PointCloud:
    """Orbbec 返回单位为 mm，这里统一转米并构造 Open3D 点云对象"""
    pcd = o3d.geometry.PointCloud()
    xyz_m = points_np[:, :3] * 0.001  # mm → m
    pcd.points = o3d.utility.Vector3dVector(xyz_m.astype(np.float32))
    if points_np.shape[1] == 6:  # 如果带 RGB
        rgb = np.clip(points_np[:, 3:6] / 255.0, 0, 1)
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32))
    return pcd


def icp_transform(source: o3d.geometry.PointCloud,
                  target: o3d.geometry.PointCloud) -> np.ndarray:
    """对低分辨率副本做点到面 ICP，返回 4×4 变换矩阵"""
    src_d = source.voxel_down_sample(VOXEL_SIZE_ICP)
    tgt_d = target.voxel_down_sample(VOXEL_SIZE_ICP)
    src_d.estimate_normals()
    tgt_d.estimate_normals()
    result = o3d.pipelines.registration.registration_icp(
        src_d, tgt_d, MAX_CORR_DIST, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result.transformation


def main():
    cam = init_camera()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("实时点云扫描 (按 Q 退出 / S 保存)", width=960, height=720)
    global_cloud = o3d.geometry.PointCloud()
    need_update_vis = False  # 首帧需要 add_geometry，后续 update

    # 键盘回调——S 保存，Q 退出
    def save_pcd(vis_obj):
        o3d.io.write_point_cloud("scan_result.ply", global_cloud)
        print("已保存 → scan_result.ply")
        return False  # 继续运行

    def quit_app(vis_obj):
        print("收到退出指令")
        raise KeyboardInterrupt

    vis.register_key_callback(ord("S"), save_pcd)
    vis.register_key_callback(ord("Q"), quit_app)

    # 捕获 Ctrl+C
    signal.signal(signal.SIGINT, lambda sig, frm: quit_app(None))

    try:
        while True:
            pts = cam.get_point_cloud(colored=True)
            if pts is None:
                continue
            frame_pcd = make_pcd_from_numpy(pts)

            if len(global_cloud.points) == 0:
                global_cloud = frame_pcd
                vis.add_geometry(global_cloud, reset_bounding_box=True)
            else:
                # ICP 配准到全局坐标系
                T = icp_transform(frame_pcd, global_cloud)
                frame_pcd.transform(T)
                # 融合并体素降采样控制规模
                global_cloud += frame_pcd
                global_cloud = global_cloud.voxel_down_sample(VOXEL_SIZE_INTEGRATE)
                need_update_vis = True

            if need_update_vis:
                vis.update_geometry(global_cloud)
            vis.poll_events()
            vis.update_renderer()

            # OpenCV 检测键盘，同步退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 保存最终点云
        o3d.io.write_point_cloud("scan_result.ply", global_cloud)
        print("已保存 → scan_result.ply")
        cam.stop()
        vis.destroy_window()


if __name__ == "__main__":
    main()

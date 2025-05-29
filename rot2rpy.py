import numpy as np
from scipy.spatial.transform import Rotation as R

# 定义绕Y轴旋转-90度和绕Z轴旋转90度
rot_x = R.from_euler('x', 90, degrees=True)

rot_y = R.from_euler('y', 0, degrees=True)
rot_z = R.from_euler('z', 0, degrees=True)

# 组合旋转矩阵
combined_rotation = rot_x * rot_y * rot_z

# 将旋转矩阵转换为RPY（以XYZ排列）
rpy = combined_rotation.as_euler('xyz', degrees=True)
print(rpy)
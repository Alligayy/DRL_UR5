import pybullet as p
import pybullet_data
import time
import os

# 连接到PyBullet仿真（GUI模式）
client = p.connect(p.GUI)

# 设置PyBullet的搜索路径，包含pybullet自带的数据
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)

# 加载地面
plane_id = p.loadURDF("plane.urdf")

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构造UR5 URDF文件的完整路径
urdf_path = os.path.join(current_dir, "urdf", "ur5_robotiq_85.urdf")

# 检查URDF文件是否存在
if not os.path.exists(urdf_path):
    raise FileNotFoundError(f"URDF file not found at {urdf_path}")
#D:\Code\Python\DRL_UR5\urdf\ur5_robot.urdf

base_pos=[0, 0, 0.62]
base_ori= p.getQuaternionFromEuler([0, 0, 0])  # 基座位置和方向

# 加载UR5机械臂
ur5_id = p.loadURDF(urdf_path, base_pos, base_ori, useFixedBase=True)# 固定基座
table_id = p.loadURDF("table/table.urdf", [0, 0.4, 0], p.getQuaternionFromEuler([0, 0, 0]))    # 加载桌子

joint_positions = []
for i in range(8):
        # 获取连杆状态 (joint/link index = i)
        link_state = p.getLinkState(ur5_id, i)
        
        # 提取空间位置 (x, y, z)
        position = link_state[0]
        joint_positions.append(position)
        print(f"Joint {i} position: {position}")


# 设置渲染模式
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
while True:
        time.sleep(1. / 240.)




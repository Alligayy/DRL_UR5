import pybullet as p
import time
from collections import namedtuple
import math
import numpy as np

class UR5Robotiq85:# UR5 with Robotiq 85 gripper
    """    UR5 with Robotiq 85 gripper class.
    This class handles the loading of the UR5 robot with a Robotiq 85 gripper in PyBullet.
    It includes methods for controlling the arm and gripper, as well as parsing joint information.
    """
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 17
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        
        # 夹爪中心点相对于基座的偏移量//
        self.gripper_center_offset = [0.0, 0.0, 0.12]  # 根据URDF文件确定的偏移量
        
        self.gripper_range = [0, 0.085]
        self.max_velocity = 10

    def load(self):
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def __parse_joint_info__(self):
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(
                jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            )

        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)]

    def __setup_mimic_joints__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id, self.id, joint_id,
                                   jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def move_gripper(self, open_length):
        """
        Control the gripper to open or close.
        :param open_length: Target width for gripper opening (0 ~ 0.085m)
        """
        open_length = max(self.gripper_range[0], min(open_length, self.gripper_range[1]))
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle)

    def move_arm_ik(self, target_pos, target_orn):
        # 计算夹爪基座需要到达的位置//
        # 将目标位置转换为基座位置//
        base_target_pos = [
            target_pos[0] + self.gripper_center_offset[0],
            target_pos[1] + self.gripper_center_offset[1],
            target_pos[2] + self.gripper_center_offset[2]
        ]
        
        joint_poses = p.calculateInverseKinematics(
            self.id, self.eef_id, base_target_pos, target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
        )
        
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i], maxVelocity=self.max_velocity)

    def get_current_ee_position(self):
        return p.getLinkState(self.id, self.eef_id)

    def orginal_position(self,robot):
        # 设置机器人初始位置
        #target_joint_positions = [2.5, -1.57, 1.57, -1.5, -1.57, 0.0]
        target_joint_positions = [2.5, -1, 1, -1.5, -1.57, 0.0]
        #target_joint_positions = [0, 0, 0, 0, 0, 0]

        for i, joint_id in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i])
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1.0/240.0)  # 240Hz simulation step
        robot.move_gripper(0.03)# 初始位置夹爪关闭，最大0.085
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1.0/240.0)

if __name__ == "__main__":
    physics_client = p.connect(p.GUI)
    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()
    robot.orginal_position(robot)
    for i in range(p.getNumJoints(robot.id)):
        print(i, p.getJointInfo(robot.id, i)[12].decode())
    
    link_state = p.getLinkState(robot.id, 17)  # 获取末端执行器状态
        # 提取位置和四元数
    position = np.array(link_state[0])
    orientation_quat = np.array(link_state[1])

    #画出目标位置小球
    ball_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.02,  # 球体半径
        rgbaColor=[1, 0, 0, 1] ) # 红色球体作为目标位置
    # 在目标位置创建一个球体
    ball_id = p.createMultiBody(
        baseMass=0,  # 质量为0，静态物体
        baseVisualShapeIndex=ball_visual,
        basePosition=position)
    while True:
        time.sleep(1. / 240.)
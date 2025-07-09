import numpy as np
import gym
from gym import spaces
import pybullet as p
import pybullet_data
import time
import math
import os

class UR5PyBulletEnv(gym.Env):
    """
    基于PyBullet的UR5机械臂轨迹规划强化学习环境
    """
    
    def __init__(self, render_mode=False, use_gui=True):
        super(UR5PyBulletEnv, self).__init__()
        
        # 环境参数
        self.use_gui = use_gui
        self.render_mode = render_mode
        self.dt = 1.0/240.0  # PyBullet默认时间步长
        self.max_steps = 1000
        self.current_step = 0
        
        # 初始化PyBullet
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # 设置物理参数
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        
        # UR5机械臂参数
        self.num_joints = 6
        self.joint_indices = list(range(self.num_joints))
        
        # 关节限制 (弧度)
        self.joint_limits_low = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_limits_high = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        
        # 关节速度限制 (rad/s)
        self.velocity_limits = np.array([3.15, 3.15, 3.15, 3.2, 3.2, 3.2])
        
        # 关节加速度限制 (rad/s²)
        self.acceleration_limits = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        
        # 状态空间: [6个关节角度, 6个关节角速度, 6个末端位姿(xyz+rpy), 3个目标位置]
        state_dim = 6 + 6 + 6 + 3  # 21维状态空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # 动作空间: 6个关节的角加速度
        self.action_space = spaces.Box(
            low=-self.acceleration_limits, 
            high=self.acceleration_limits, 
            dtype=np.float32
        )
        
        # 末端执行器链接索引
        self.end_effector_index = 7  # UR5的末端执行器链接索引
        
        # 工作空间限制
        self.workspace_limits = {
            'x': [-1.0, 1.0],
            'y': [-1.0, 1.0],
            'z': [0.0, 1.5]
        }
        
        # 奖励参数
        self.position_weight = 1.0
        self.velocity_weight = 0.1
        self.acceleration_weight = 0.01
        self.success_reward = 100.0
        self.distance_threshold = 0.05  # 成功距离阈值
        self.collision_penalty = -10.0
        
        # 目标位置
        self.target_position = np.zeros(3)
        self.target_sphere_id = None
        
        # 加载环境
        self._load_environment()
    
    def _load_environment(self):
        """
        加载环境和机械臂
        """
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
        
        # 加载UR5机械臂
        # 如果没有UR5的URDF文件，可以使用Kuka机械臂作为替代
        try:
            # 尝试加载UR5 URDF文件
            self.robot_id = p.loadURDF("ur5/ur5.urdf", [0, 0, 0], useFixedBase=True)
        except:
            # 如果没有UR5，使用Kuka机械臂
            print("Warning: UR5 URDF not found, using Kuka arm instead")
            self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
            self.end_effector_index = 6  # Kuka的末端执行器索引
            self.num_joints = 7  # Kuka有7个关节
            self.joint_indices = list(range(self.num_joints))
            # 更新关节限制
            self.joint_limits_low = np.array([-2.97, -2.09, -2.97, -2.09, -2.97, -2.09, -3.05])
            self.joint_limits_high = np.array([2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05])
            self.velocity_limits = np.array([1.48, 1.48, 1.75, 1.31, 2.27, 2.36, 2.36])
            self.acceleration_limits = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
            
            # 更新状态空间
            state_dim = self.num_joints * 2 + 6 + 3  # 关节角度+关节速度+末端位姿+目标位置
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-self.acceleration_limits, 
                high=self.acceleration_limits, 
                dtype=np.float32
            )
        
        # 获取机械臂信息
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"Robot loaded with {self.num_joints} joints")
        
        # 设置关节阻尼
        for i in range(self.num_joints):
            p.changeDynamics(self.robot_id, i, jointDamping=0.1)
        
        # 设置摄像头视角
        if self.use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.5]
            )
    
    def _create_target_sphere(self, position):
        """
        创建目标球体
        """
        if self.target_sphere_id is not None:
            p.removeBody(self.target_sphere_id)
        
        # 创建球体碰撞形状
        sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
        sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
        
        # 创建目标球体
        self.target_sphere_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=position
        )
    
    def _generate_random_target(self):
        """
        生成随机目标位置
        """
        # 在工作空间内生成随机目标
        x = np.random.uniform(self.workspace_limits['x'][0], self.workspace_limits['x'][1])
        y = np.random.uniform(self.workspace_limits['y'][0], self.workspace_limits['y'][1])
        z = np.random.uniform(self.workspace_limits['z'][0], self.workspace_limits['z'][1])
        
        # 确保目标在机械臂可达范围内
        max_reach = 0.8  # 大致的最大可达距离
        distance = np.sqrt(x**2 + y**2 + z**2)
        if distance > max_reach:
            scale = max_reach / distance
            x *= scale
            y *= scale
            z *= scale
        
        return np.array([x, y, z])
    
    def _get_joint_states(self):
        """
        获取关节状态
        """
        joint_states = p.getJointStates(self.robot_id, self.joint_indices[:self.num_joints])
        joint_angles = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        return joint_angles, joint_velocities
    
    def _get_end_effector_pose(self):
        """
        获取末端执行器位姿
        """
        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        position = np.array(link_state[0])
        orientation_quat = np.array(link_state[1])
        
        # 将四元数转换为欧拉角
        orientation_euler = p.getEulerFromQuaternion(orientation_quat)
        
        return position, np.array(orientation_euler)
    
    def _get_state(self):
        """
        获取当前状态
        """
        joint_angles, joint_velocities = self._get_joint_states()
        end_effector_pos, end_effector_orient = self._get_end_effector_pose()
        
        # 只使用前6个关节（如果是7自由度机械臂）
        if self.num_joints > 6:
            joint_angles = joint_angles[:6]
            joint_velocities = joint_velocities[:6]
        
        state = np.concatenate([
            joint_angles,               # 6个关节角度
            joint_velocities,           # 6个关节角速度
            end_effector_pos,          # 末端位置 (3D)
            end_effector_orient,       # 末端姿态 (RPY)
            self.target_position       # 目标位置 (3D)
        ])
        
        return state.astype(np.float32)
    
    def _check_collision(self):
        """
        检查碰撞
        """
        # 获取碰撞信息
        contact_points = p.getContactPoints(bodyA=self.robot_id)
        
        # 检查是否与地面以外的物体碰撞
        for contact in contact_points:
            if contact[2] != self.plane_id:  # 不是与地面的碰撞
                return True
        
        return False
    
    def _calculate_reward(self, action):
        """
        计算奖励
        """
        end_effector_pos, _ = self._get_end_effector_pose()
        
        # 计算距离奖励
        distance = np.linalg.norm(end_effector_pos - self.target_position)
        distance_reward = -distance * self.position_weight
        
        # 计算速度惩罚
        _, joint_velocities = self._get_joint_states()
        velocity_penalty = -np.sum(np.abs(joint_velocities)) * self.velocity_weight
        
        # 计算加速度惩罚
        acceleration_penalty = -np.sum(np.abs(action)) * self.acceleration_weight
        
        # 成功奖励
        success_reward = 0
        if distance < self.distance_threshold:
            success_reward = self.success_reward
        
        # 碰撞惩罚
        collision_penalty = 0
        if self._check_collision():
            collision_penalty = self.collision_penalty
        
        total_reward = distance_reward + velocity_penalty + acceleration_penalty + success_reward + collision_penalty
        
        return total_reward, distance < self.distance_threshold, self._check_collision()
    
    def step(self, action):
        """
        执行一步动作
        """
        # 限制动作范围
        action = np.clip(action, -self.acceleration_limits, self.acceleration_limits)
        
        # 获取当前关节状态
        joint_angles, joint_velocities = self._get_joint_states()
        
        # 只使用前6个关节的动作
        if len(action) > 6:
            action = action[:6]
        if len(joint_angles) > 6:
            joint_angles = joint_angles[:6]
            joint_velocities = joint_velocities[:6]
        
        # 使用动作（加速度）更新关节速度
        new_joint_velocities = joint_velocities + action * self.dt
        
        # 限制关节速度
        new_joint_velocities = np.clip(new_joint_velocities, -self.velocity_limits[:len(new_joint_velocities)], self.velocity_limits[:len(new_joint_velocities)])
        
        # 计算新的关节角度
        new_joint_angles = joint_angles + new_joint_velocities * self.dt
        
        # 限制关节角度
        new_joint_angles = np.clip(new_joint_angles, self.joint_limits_low[:len(new_joint_angles)], self.joint_limits_high[:len(new_joint_angles)])
        
        # 设置关节目标位置
        for i in range(len(new_joint_angles)):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=new_joint_angles[i],
                targetVelocity=new_joint_velocities[i]
            )
        
        # 执行物理仿真
        p.stepSimulation()
        
        # 获取新状态
        state = self._get_state()
        
        # 计算奖励
        reward, success, collision = self._calculate_reward(action)
        
        # 检查终止条件
        self.current_step += 1
        done = success or collision or self.current_step >= self.max_steps
        
        # 信息字典
        info = {
            'success': success,
            'collision': collision,
            'distance': np.linalg.norm(self._get_end_effector_pose()[0] - self.target_position),
            'step': self.current_step
        }
        
        return state, reward, done, info
    
    def reset(self):
        """
        重置环境
        """
        # 重置步数
        self.current_step = 0
        
        # 生成新的目标位置
        self.target_position = self._generate_random_target()
        self._create_target_sphere(self.target_position)
        
        # 重置机械臂到初始位置
        initial_joint_angles = np.random.uniform(
            self.joint_limits_low[:6] * 0.5,
            self.joint_limits_high[:6] * 0.5
        )
        
        for i in range(min(6, self.num_joints)):
            p.resetJointState(self.robot_id, i, initial_joint_angles[i])
        
        # 执行几步仿真以稳定系统
        for _ in range(10):
            p.stepSimulation()
        
        # 返回初始状态
        return self._get_state()
    
    def render(self, mode='human'):
        """
        渲染环境
        """
        if mode == 'human' and self.use_gui:
            # GUI模式下自动渲染
            pass
        elif mode == 'rgb_array':
            # 获取摄像头图像
            width, height = 320, 240
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.5],
                distance=1.5,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=width/height,
                nearVal=0.1,
                farVal=100.0
            )
            
            (_, _, px, _, _) = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (height, width, 4))
            rgb_array = rgb_array[:, :, :3]
            return rgb_array
    
    def close(self):
        """
        关闭环境
        """
        p.disconnect(self.physics_client)

# 使用示例
if __name__ == "__main__":
    # 创建环境
    env = UR5PyBulletEnv(use_gui=True)
    
    # 测试环境
    for episode in range(5):
        state = env.reset()
        episode_reward = 0
        
        print(f"Episode {episode + 1} started")
        
        for step in range(200):
            # 随机动作
            action = env.action_space.sample()
            
            # 执行动作
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # 渲染
            env.render()
            time.sleep(0.01)
            
            if done:
                print(f"Episode {episode + 1} finished after {step + 1} steps")
                print(f"Episode reward: {episode_reward:.2f}")
                print(f"Success: {info['success']}, Collision: {info['collision']}")
                print(f"Final distance: {info['distance']:.3f}")
                break
        
        print("-" * 50)
    
    env.close()
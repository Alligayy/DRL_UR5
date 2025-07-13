import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import math

from ur5_robot import UR5Robotiq85  # Import the UR5Robotiq85 class from ur5_robot.py

class UR5RobotiqEnv(gym.Env):
    def __init__(self, reward_type='dense'):
        super(UR5RobotiqEnv, self).__init__()

         # 连接PyBullet
        self.physics_client = p.connect(p.GUI) # 使用GUI模式连接
        #self.physics_client = p.connect(p.DIRECT)  # 使用DIRECT模式连接
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # 仿真时间步长1/240s
        p.setTimeStep(1.0/240.0)


        self.dt = 1.0/240.0# PyBullet默认时间步长，用于仿真
        self.time_step = 0.1  #时间步长
        # 设置最大步数
        self.max_steps = 100
        self.current_step = 0
        self.reward_type = reward_type # 奖励类型， 稀疏奖励'sparse' or 'dense'稠密奖励
        
        # 关节角度限制 (rad)
        self.joint_limits = np.full(6, np.pi) # 关节角度范围为[-π, π]
        # 关节速度限制 (rad/s)
        self.velocity_limits = np.full(6, 5) # 关节速度范围为[-5, 5] rad/s
        # 关节加速度限制 (rad/s²)
        self.acceleration_limits = np.full(6, 5) # 关节加速度范围为[-5, 5] rad/s²

        # 动作空间范围
        action_dim = 6 
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(6,), dtype=np.float64)    #6个关节的加速度

        # 状态空间范围
        state_dim = 6 + 6 + 3 #15维状态空间
        self.observation_space = spaces.Dict({"observation": spaces.Box(
            low=np.concatenate((
                np.full(6, -np.pi), # 6个关节角度
                np.full(6, -5),     # 6个关节速度
                np.full(3, -np.inf), # 当前时刻障碍物位置
                # np.full(3, -np.inf), # 关节1位置
                # np.full(3, -np.inf), # 关节2位置
                # np.full(3, -np.inf), # 关节3位置
                # np.full(3, -np.inf), # 关节4位置
                # np.full(3, -np.inf), # 关节5位置
                # np.full(3, -np.inf), # 关节6位置
                #np.full(3, -5), # 末端执行器位置范围
                # np.full(3, -np.inf), # 上一时刻障碍物位置
                # np.full(3, -np.inf), # 当前时刻障碍物位置
                #np.full(3, -1), # 目标位置范围  
            )),
            high=np.concatenate((
                np.full(6, np.pi), # 6个关节角度
                np.full(6, 5),     # 6个关节速度
                np.full(3, np.inf), # 当前时刻障碍物位置
                # np.full(3, np.inf), # 关节1位置
                # np.full(3, np.inf), # 关节2位置
                # np.full(3, np.inf), # 关节3位置
                # np.full(3, np.inf), # 关节4位置
                # np.full(3, np.inf), # 关节5位置
                # np.full(3, np.inf), # 关节6位置
                #np.full(3, 5), # 末端执行器位置范围
                # np.full(3, np.inf), # 上一时刻障碍物位置
                # np.full(3, np.inf), # 当前时刻障碍物位置
                #np.full(3, 1), # 目标位置范围
            )),dtype=np.float64),
            
            "achieved_goal": spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float64),# 当前机械臂末端位置

            "desired_goal": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64)# 目标位置
        })
        # 加载环境物体
        self.plane_id = p.loadURDF("plane.urdf")    # 加载地面
        self.table_id = p.loadURDF("table/table.urdf", [0, 0.4, 0], p.getQuaternionFromEuler([0, 0, 0]))    # 加载桌子
        self.ball_id = None  # 用于保存目标位置的小球ID
        self.obstacle_id = None     # 用于保存障碍物的小球ID
        # 加载UR5机器人
        self.robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        # 初始化需要到达的目标位置
        self.target_position = None
        self.obstacle_position = None
        
        # 用于保存目标位置的debug对象引用
        self.target_debug_points = []
        
    def _generate_random_target(self): 
        """
        生成随机目标位置
        """
        # 随机生成目标位置
        x_range = [0.4, 0.5]#目标点生成区域
        y_range = [0.3, 0.5]
        z_range = [0.65, 0.8]
        target_pos = [
            np.random.uniform(x_range[0], x_range[1]),  
            np.random.uniform(y_range[0], y_range[1]),  
            np.random.uniform(z_range[0], z_range[1])]

        return np.array(target_pos)
    
    def _generate_random_obstacle(self): 
        """
        生成随机障碍物位置
        """
        # 随机生成障碍物位置
        x_range = [-0.4, 0.3]#障碍物生成区域
        y_range = [0.3, 0.8]
        z = 0.72
        obstacle_pos = [
            np.random.uniform(x_range[0], x_range[1]),  
            np.random.uniform(y_range[0], y_range[1]),  
            z]

        return np.array(obstacle_pos)
    
    def _get_joint_states(self):
        """
        获取关节状态
        """
        joint_states = p.getJointStates(self.robot.id, self.robot.arm_controllable_joints)# 获取关节状态
        # 提取关节角度和速度
        joint_angles = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        return joint_angles, joint_velocities
    
    def _get_end_effector_pose(self):
        """
        获取末端执行器位置
        """
        link_state = p.getLinkState(self.robot.id, self.robot.eef_id)  # 获取末端执行器状态
        # 提取位置和四元数
        position = np.array(link_state[0])
        orientation_quat = np.array(link_state[1])
        # # 将四元数转换为欧拉角
        # orientation_euler = p.getEulerFromQuaternion(orientation_quat)
        
        return position, orientation_quat #, np.array(orientation_euler) 

    def _check_obstacle_collision(self, achieved_goal, obstacle_position):
        """
        检查机械臂末端是否与障碍物碰撞
        """
        # 
        dis_oe = np.linalg.norm(achieved_goal - obstacle_position, axis=-1)
        if dis_oe < 0.1:  # 如果距离小于0.1，则认为发生碰撞
            return True
        else:
            return False

    
    def _get_obs(self):
        """
        获取当前状态空间
        """
        joint_angles, joint_velocities = self._get_joint_states()
        #end_effector_pos, end_effector_orient = self._get_end_effector_pose()
        end_effector_pos, _ = self._get_end_effector_pose()
        
        # 只使用前6个关节（如果是7自由度机械臂）
        # if self.num_joints > 6:
        #     joint_angles = joint_angles[:6]
        #     joint_velocities = joint_velocities[:6]
        
        observation = np.concatenate([
            joint_angles,               # 6个关节角度
            joint_velocities,           # 6个关节角速度
            self.obstacle_position      # 障碍物位置
            #end_effector_pos,           # 末端位置
            #end_effector_orient,        # 末端姿态 (RPY)
            #self.target_position        # 目标位置
        ]).astype(np.float64)

        # 当前达到的目标：末端执行器位置
        achieved_goal = end_effector_pos.astype(np.float64)
        
        # 期望的目标位置
        desired_goal = self.target_position.astype(np.float64)
        
        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }
    
    def _is_success(self, achieved_goal, desired_goal):
        """
        判断是否成功到达目标
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < self.distance_threshold

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """
        计算到达奖励函数（HER兼容）
        """
        # 计算欧几里得距离
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        
        if self.reward_type == 'sparse':
            # 稀疏奖励：只有成功时给予0，失败时给予-1
            return -(distance > self.distance_threshold).astype(np.float64)
        else:
            # 稠密奖励：基于距离的连续奖励
            return -distance*5

    def compute_obstacle_reward(self, achieved_goal, desired_goal, obstacle_position, ):
        """
        计算机械臂避障奖励函数
        """
        # 计算欧几里得距离
        dis_ge = np.linalg.norm(achieved_goal - desired_goal, axis=-1) # 目标位置到当前位置的距离
        dis_oe = np.linalg.norm(achieved_goal - obstacle_position, axis=-1) - 0.1 # 障碍物位置到当前位置的距离，需减去障碍物半径
        
        # 计算避障奖励
        if dis_oe > 0.3:    # 安全区
            r_obstacle = r_safe = math.exp(-dis_ge)-math.exp(-dis_oe)
        elif 0 < dis_oe < 0.1:  # 危险区
            r_obstacle = r_danger = math.log(dis_oe)  
        elif dis_oe <= 0:       # 发生碰撞，给个极小的负奖励
            r_obstacle = -500
        else:               # 警告区
            r_safe = math.exp(-dis_ge)-math.exp(-dis_oe)
            r_danger = math.log(dis_oe)
            r_obstacle = ((dis_oe - 0.1)/(0.3-0.1))*r_safe + ((0.3 - dis_oe)/(0.3-0.1))*r_danger
        return r_obstacle

    def compute_slow_down_reward(self, joint_velocities):
        """
        计算接近目标时减速奖励函数
        """
        return math.exp(-(np.linalg.norm(joint_velocities)))  
    def compute_smooth_reward(self, joint_velocities, new_joint_velocities):
        """
        计算平滑度奖励函数
        """
        # 计算速度之差
        distance = np.linalg.norm(new_joint_velocities - joint_velocities, axis=-1)
        
        return math.exp(-distance)
    def cal_distance(self):
        current_pos, _ = self._get_end_effector_pose()
        distance = np.linalg.norm(current_pos - self.target_position)
        return distance

    def cal_arm_ik(self, target_pos, target_orn):#计算机械臂夹爪到目标位姿的逆运动学
        # 计算夹爪基座需要到达的位置//
        # 将目标位置转换为基座位置//
        #base_target_pos = target_pos + self.robot.gripper_center_offset
        base_target_pos = [
            target_pos[0] + self.robot.gripper_center_offset[0],
            target_pos[1] + self.robot.gripper_center_offset[1],
            target_pos[2] + self.robot.gripper_center_offset[2]
        ]
        #可控关节的逆运动学计算
        #存储可控关节
        controllable_joints_angles = []#存储可控关节

        joints_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.robot.id, 
            endEffectorLinkIndex=self.robot.eef_id, 
            targetPosition = base_target_pos, 
            targetOrientation = target_orn,
            lowerLimits = self.robot.arm_lower_limits,
            upperLimits = self.robot.arm_upper_limits,
            jointRanges = self.robot.arm_joint_ranges,
            restPoses = self.robot.arm_rest_poses,)
        
        #存储可控关节角度
        # for i in self.robot.arm_controllable_joints:
        #     controllable_joints_angles.append(joints_angles[i])

        #可调试移动机械臂到目标位置
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            # p.setJointMotorControl2(
            #     bodyIndex=self.robot.id,
            #     jointIndex=joint_id,
            #     controlMode=p.POSITION_CONTROL,
            #     targetPosition=joints_angles[i],
            #     maxVelocity=self.robot.max_velocity
            # )
            controllable_joints_angles.append(joints_angles[i])  # 更新可控关节角度
            
        # 执行仿真步
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.dt)
        
        return controllable_joints_angles  # 返回计算得到的关节角度
    def create_custom_sphere(self,position, color, radius, mass=0):#创建自定义球体
        """
        创建自定义球体
        :param radius: 球体半径
        :param mass: 球体质量
        :param position: 球体位置 [x, y, z]
        :param color: 球体颜色 [r, g, b, a]
        :return: 球体对象ID
        """
        collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        sphere_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )
        return sphere_id

    def set_sphere_velocity(self, sphere_id, linear_velocity=[0.02, 0, 0]):
        """
        给指定的小球设置线速度
        :param sphere_id: 小球对象ID
        :param linear_velocity: 线速度向量 [vx, vy, vz]
        """
        p.resetBaseVelocity(
            objectUniqueId=sphere_id,
            linearVelocity=linear_velocity
        )


    def reset(self, seed=None, options=None):
        """
        重置环境
        """
        self.current_step = 0# 重置步数
        self.robot.orginal_position(self.robot) #机械臂回起点
        # 执行仿真步
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.dt)
        self.distance_threshold = 0.05  # 定义成功到达目标的距离阈值

        
        # 清除之前的目标位置小球
        if self.ball_id is not None:
            p.removeBody(self.ball_id)
        # 清除之前的障碍物
        if self.obstacle_id is not None:
            p.removeBody(self.obstacle_id)

        # 生成随机目标位置
        self.target_position = self._generate_random_target()
        self.obstacle_position = self._generate_random_obstacle() # 生成随机障碍物位置
        
        #创建目标位置小球
        
        self.ball_id = self.create_custom_sphere(
            position=self.target_position, 
            radius=0.02, 
            mass=0, 
            color=[1, 0, 0, 1])
       
        #创建障碍物球体
        self.obstacle_id = self.create_custom_sphere(
            position=self.obstacle_position, 
            radius=0.1, 
            mass=0, 
            color=[0, 0, 0.3, 1])   
        
        # 设置障碍物速度
        self.set_sphere_velocity( self.obstacle_id, linear_velocity=[0.02, 0, 0])  # 设置障碍物的线速度
        

        # 计算目标关节角度
        _, self.target_orn = self._get_end_effector_pose()  # 获取末端执行器的当前姿态
        self.target_joints = self.cal_arm_ik(target_pos=self.target_position, target_orn=self.target_orn) #姿态默认
        
        #调试代码
        '''
        #self.robot.move_arm_ik(target_pos=self.target_position, target_orn=target_orn) #姿态默认
        # 执行仿真步
        # for _ in range(100):
        #     p.stepSimulation()
        #     time.sleep(self.dt)
        # for i, joint_id in enumerate(self.robot.arm_controllable_joints):
        #     p.setJointMotorControl2(
        #         bodyIndex=self.robot.id,
        #         jointIndex=joint_id,
        #         controlMode=p.POSITION_CONTROL,
        #         targetPosition=self.target_joints[i],
        #         maxVelocity=self.robot.max_velocity
        #     )
            
        # # 执行仿真步
        # for _ in range(100):
        #     p.stepSimulation()
        #     time.sleep(self.dt)

        pause = 1  # 暂停1秒以确保机械臂到达目标位置
        time.sleep(pause)  # 暂停1秒
        self.robot.orginal_position(self.robot) #机械臂回起点
        # 执行仿真步
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.dt)'''
        
        # 获取当前状态
        obs = self._get_obs()
        # 返回观测值
        info={} #用于返回额外调试信息，推荐添加
        return obs, info

    def step(self, action):
        """
        执行一步动作
        """
        # 获取当前关节状态
        joint_angles, joint_velocities = self._get_joint_states()

        # 限制动作范围
        action = np.clip(action, -self.acceleration_limits, self.acceleration_limits)

        # 只使用前6个关节的动作
        if len(action) > 6:
            action = action[:6]
        if len(joint_angles) > 6:
            joint_angles = joint_angles[:6]
            joint_velocities = joint_velocities[:6]
        
        # 计算新的关节速度
        # 使用动作（加速度）更新关节速度
        new_joint_velocities = joint_velocities + action * self.time_step
        # 限制关节速度
        new_joint_velocities = np.clip(new_joint_velocities, -self.velocity_limits[:len(new_joint_velocities)], self.velocity_limits[:len(new_joint_velocities)])
        # 计算新的关节角度
        new_joint_angles = joint_angles + new_joint_velocities * self.time_step
        #逆运动学角度修正
        distance = self.cal_distance()  # 计算当前末端执行器与目标位置的距离
        if distance < 0.3: # 如果距离小于0.3，则加入逆运动学修正
            new_joint_angles += (self.target_joints - joint_angles) * self.time_step  # 添加逆运动学目标关节角度的偏差

        # 限制关节角度
        new_joint_angles = np.clip(new_joint_angles, -self.joint_limits[:len(new_joint_angles)], self.joint_limits[:len(new_joint_angles)])

        # 设置新的关节角度
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            p.setJointMotorControl2(
                bodyIndex=self.robot.id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=new_joint_angles[i],
                force=500,  # 设置一个合适的力
                positionGain=0.1,  # 设置位置增益
                velocityGain=0.1   # 设置速度增益
            )
        
        # new_obstacle_position = self._obstacle_random_move(self.obstacle_position)  # 障碍物随机移动
        new_obstacle_position, _ = p.getBasePositionAndOrientation(self.obstacle_id)  # 获取当前障碍物位置
        x_range = [-0.4, 0.3]#障碍物生成区域
        y_range = [0.3, 0.8]
        if new_obstacle_position[0] < x_range[0] or new_obstacle_position[0] > x_range[1] or \
           new_obstacle_position[1] < y_range[0] or new_obstacle_position[1] > y_range[1]:
            # 如果障碍物位置超出范围，则重新生成
            self.set_sphere_velocity(self.obstacle_id, linear_velocity=[-0.02, 0, 0])  # 设置障碍物的线速度
        
        # self.obstacle_position = new_obstacle_position

        # # 清除之前的障碍物
        # if self.obstacle_id is not None:
        #     p.removeBody(self.obstacle_id)
        # #创建障碍物球体
        # obstacle_visual = p.createVisualShape(
        # shapeType=p.GEOM_SPHERE,
        # radius=0.1,  # 球体半径
        # rgbaColor=[0, 0, 0.3, 1] ) # 蓝色球体作为障碍物
        # # 在目标位置创建一个球体
        # self.obstacle_id = p.createMultiBody(
        # baseMass=0,  # 质量为0，静态物体
        # baseVisualShapeIndex=obstacle_visual,
        # basePosition=new_obstacle_position)

        # 执行仿真步
        for _ in range(100):
            p.stepSimulation()
            
        # 获取新状态
        state = self._get_obs()

        # 检查终止条件:成功/超时/碰撞
        self.current_step += 1
        # 检查成功
        success = self._is_success(state['achieved_goal'], state['desired_goal'])
        # 检查碰撞
        collision = self._check_obstacle_collision(state['achieved_goal'], new_obstacle_position)
        done = success or self.current_step >= self.max_steps or collision    
        
        # 信息字典
        info = {
            'is_success': success,
            'collision': collision,
            'distance': np.linalg.norm(state['achieved_goal'] - state['desired_goal']),
            'step': self.current_step
        }

        # 计算奖励
        reward_goal = self.compute_reward(state['achieved_goal'], state['desired_goal'], info) 
        reward_obstacle = self.compute_obstacle_reward(state['achieved_goal'], state['desired_goal'], self.obstacle_position)
        
        # 计算减速奖励
        if distance < 0.2:  # 如果距离小于0.2，则加入减速奖励
            reward_slow_down = self.compute_slow_down_reward(new_joint_velocities)
        else:
            reward_slow_down = 0

        reward_smooth = self.compute_smooth_reward(joint_velocities, new_joint_velocities) # 计算平滑度奖励
        # 计算总奖励
        reward = reward_goal + reward_obstacle + reward_slow_down + reward_smooth
        # 如果成功到达目标，奖励+400
        if success:
            reward += 400 + (self.max_steps-self.current_step)  # 成功奖励+400，步数越小奖励越高
        
        if self.current_step >= self.max_steps:
            reward -= 100
        
        # 输出调试信息
        distance_to_target = info['distance']
        print(f"step:{self.current_step}\t",f"reward:{reward}")
        print(f"Distance difference: {distance_to_target}\n")
        
        truncated = False# 无外部终止，即在这个环境中没有截断条件
        return state, reward, done, truncated, info

    def close(self):
        p.disconnect()





     

    
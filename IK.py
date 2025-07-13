import numpy as np

# D-H 参数
dh_params = [
    [0, 0.089,  0,  np.pi / 2],
    [0, 0,   -0.425,          0],
    [0, 0,    0.236,          0],
    [0, 0.109,  0,  np.pi / 2],
    [0, 0.094,  0, -np.pi / 2],
    [0, 0.082,  0,          0]
]

def transformation_matrix(theta, d, a, alpha):
    """生成单个关节的变换矩阵"""
    return np.array([
        [np.cos(theta),     -np.sin(theta) * np.cos(alpha),     np.sin(theta) * np.sin(alpha),     a * np.cos(theta)],
        [np.sin(theta),      np.cos(theta) * np.cos(alpha),    -np.cos(theta) * np.sin(alpha),     a * np.sin(theta)],
        [            0,      np.sin(alpha),                     np.cos(alpha),                     d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(thetas):
    """正向运动学：计算末端执行器的位置和姿态"""
    T = np.eye(4)
    for i, theta in enumerate(thetas):
        T_i = transformation_matrix(theta + dh_params[i][0], dh_params[i][1], dh_params[i][2], dh_params[i][3])
        T = T @ T_i
    return T

def jacobian(thetas):
    """计算雅可比矩阵"""
    J = np.zeros((6, 6))
    T = np.eye(4)
    z_prev = np.array([0, 0, 1])
    o_prev = np.array([0, 0, 0])
    
    for i in range(6):
        T_i = transformation_matrix(thetas[i] + dh_params[i][0], dh_params[i][1], dh_params[i][2], dh_params[i][3])
        T = T @ T_i
        
        z = T[:3, 2]
        o = T[:3, 3]
        
        J[:3, i] = np.cross(z_prev, o - o_prev)
        J[3:, i] = z_prev
        
        z_prev = z
        o_prev = o
    
    return J

def newton_raphson(target_pose, initial_guess, max_iter=100, tol=1e-6):
    """牛顿下山法求解逆运动学"""
    thetas = initial_guess
    for _ in range(max_iter):
        current_pose = forward_kinematics(thetas)
        error = target_pose - current_pose
        
        # 只考虑位置误差
        pos_error = error[:3, 3]
        if np.linalg.norm(pos_error) < tol:
            return thetas
        
        J = jacobian(thetas)
        delta_thetas = np.linalg.lstsq(J[:3], -pos_error, rcond=None)[0]
        
        thetas += delta_thetas
    
    return thetas

# 示例用法
target_pose = np.array([
    [1, 0, 0, 500],
    [0, 1, 0, 0],
    [0, 0, 1, 500],
    [0, 0, 0, 1]
])

initial_guess = np.array([0, 0, 0, 0, 0, 0])
solution = newton_raphson(target_pose, initial_guess)
print("Solution:", solution)
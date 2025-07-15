from env import  UR5RobotiqEnv
from stable_baselines3 import PPO, SAC,A2C
import time
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
def test():
    # 初始化UR5机器人环境
    env = UR5RobotiqEnv()  # 创建UR5机器人环境实例
    
    # 选择要使用的算法 (SAC, PPO, or A2C)
    algo_name = "SAC"  # 设置要使用的算法名称， "PPO", "SAC", 或 "A2C"
    if algo_name == "PPO":
        model = PPO("MultiInputPolicy", env, verbose=1)  
    elif algo_name == "SAC":
        model = SAC("MultiInputPolicy",
                    env,
                    verbose=1,
                    replay_buffer_class= HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                        goal_selection_strategy="future",  # 或者"final"
                        n_sampled_goal=4,                  # 每个transition采样的额外目标数
                        )
                )  
    elif algo_name == "A2C":
        model = A2C("MultiInputPolicy", env, verbose=1)  
    else:
        raise ValueError("Invalid algorithm name. Please choose 'PPO', 'SAC', or 'A2C'.")  

    # 加载预训练模型
    model = model.load(f"./models/ur_robot_{algo_name.lower()}_100000_steps", env=env)

    # 重置环境以开始测试
    obs, info = env.reset()

    # 开始测试循环
    while True:
        # 获取当前观测值
        action, _ = model.predict(obs, deterministic=True)
        
        
        time.sleep(1/240)
        
        # 执行动作并获取下一个观测值、奖励、终止状态和截断状态
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 检测是否终止
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == '__main__':
    test()  

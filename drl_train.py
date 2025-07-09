from env import  UR5RobotiqEnv
import gymnasium as gym
from stable_baselines3 import PPO, SAC,A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import tensorboard

from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
def train():

    # 日志地址
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir) 

    # 初始化环境
    env = UR5RobotiqEnv()  
    env = Monitor(env, log_dir)  
    #归一化
    #env = gym.wrappers.NormalizeObservation(env)  # 归一化观测空间
    #env = gym.wrappers.NormalizeReward(env)  # 归一化奖励
   

    # 选择训练的算法 (SAC, PPO, or A2C)
    algo_name = "SAC"  
    if algo_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)  # "MlpPolicy" 多层感知机
    elif algo_name == "SAC":
        #model = SAC("MlpPolicy", env, verbose=1)  # verbose=1 表示显示基本的训练信息
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            #batch_size=512,  # 设置批量大小
            learning_starts=200, # 设置开始更新网络的步数
            replay_buffer_class= HerReplayBuffer,
            replay_buffer_kwargs=dict(
                goal_selection_strategy="future",  # 或者"final"
                n_sampled_goal=4,                  # 每个transition采样的额外目标数
            ),
            #使用tensorboard记录训练过程
            tensorboard_log="./tensorboard_SAC_logs",  # 设置TensorBoard日志目录
        )

    elif algo_name == "A2C":
        model = A2C("MlpPolicy", env, verbose=1)  
    else:
        raise ValueError("Invalid algorithm name. Please choose 'PPO', 'SAC', or 'A2C'.")  # Raise an error if an invalid algorithm is specified

    # Set up a checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models", name_prefix=f"ur_robot_{algo_name.lower()}")

    # Train the model
    model.learn(total_timesteps=10000, callback=checkpoint_callback)

    # Save the last model
    model.save(f"ur_robot_last_{algo_name.lower()}")

# Function to smooth the data using a moving average
def smooth(data, window_size=10):
    """Apply weighted moving average smoothing to the data"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Function to plot the training reward data
def plot_reward_data():
    # Set the directory for storing Monitor log files
    log_dir = "./logs"
    files = {
        "PPO": "monitor_ppo.csv",
        "A2C": "monitor_a2c.csv",
        "SAC": "monitor_sac.csv"
    }

    # Create a plot for the rewards over time
    plt.figure(figsize=(12, 6))

    # Iterate through the files for each algorithm
    for label, file_name in files.items():
        monitor_file = os.path.join(log_dir, file_name)
        
        # Read the Monitor log data (skip the first row of comments)
        data = pd.read_csv(monitor_file, skiprows=1)
        
        # Calculate the cumulative timestep
        data['timestep'] = np.cumsum(data['l'])  # Accumulate the timestep, as global timestep

        data['timestep'] -= data['timestep'].iloc[0]  
        
        # Apply smoothing to the reward data
        smoothed_rewards = smooth(data['r'])
        
        # Plot the smoothed rewards against the timestep
        plt.plot(
            data['timestep'][:len(smoothed_rewards)], 
            smoothed_rewards, 
            label=label
        )
    
    # Add labels and title to the plot
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    
    # Add legend to the plot
    plt.legend()
    plt.grid()  # Display grid lines
    plt.show()  # Show the plot

if __name__ == '__main__':
    train()  # Run the main function to train the model
    #plot_reward_data()  # Plot the training reward data
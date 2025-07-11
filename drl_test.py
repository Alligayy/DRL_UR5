from env import  UR5RobotiqEnv
from stable_baselines3 import PPO, SAC,A2C
import time

def test():
    # Initialize the environment
    env = UR5RobotiqEnv()  # Instantiate the custom driving environment
    
    # Predefine the algorithm to use (PPO, SAC, or A2C)
    algo_name = "SAC"  # Set the algorithm to use (SAC, PPO, or A2C)
    if algo_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)  # Choose PPO if specified
    elif algo_name == "SAC":
        model = SAC("MlpPolicy", env, verbose=1)  # Choose SAC if specified
    elif algo_name == "A2C":
        model = A2C("MlpPolicy", env, verbose=1)  # Choose A2C if specified
    else:
        raise ValueError("Invalid algorithm name. Please choose 'PPO', 'SAC', or 'A2C'.")  # Raise an error if an invalid algorithm is specified

    # Load the trained model
    model = model.load(f"./models/ur_robot_{algo_name.lower()}_100000_steps")

    # Reset the environment and get the initial observation
    obs, info = env.reset()

    # Test the trained model
    while True:
        # Use the model to predict the next action based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        
        # Sleep for 1/300 seconds to control the speed of the simulation
        time.sleep(1/300)
        
        # Execute the action and get the next state
        obs, reward, terminated, truncated, info = env.step(action)
        
        # If the environment reaches termination or truncation, reset it
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == '__main__':
    test()  #test the model

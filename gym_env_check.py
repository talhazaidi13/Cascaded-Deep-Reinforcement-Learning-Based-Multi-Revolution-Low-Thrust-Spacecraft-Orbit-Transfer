from stable_baselines3.common.env_checker import check_env
from Spacecraft_Env import Spacecraft_Env


env = Spacecraft_Env()
# It will check your custom environment and output additional warnings if needed
check_env(env)
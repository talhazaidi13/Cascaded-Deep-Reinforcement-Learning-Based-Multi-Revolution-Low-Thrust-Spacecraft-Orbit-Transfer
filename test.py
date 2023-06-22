import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import os
from Spacecraft_Env import  Spacecraft_Env
import time
#from enviornment import Enviornment
import numpy as np
import tensorflow as tf 
import os.path
import time
from datetime import date
from config import args
from scenerios import cases

tf.debugging.set_log_device_placement(True)
import argparse
from config import args

chosen_case = cases[args.case]
model_path =  chosen_case['weights_path'][0]
current_dir = os.getcwd()
models_dir = os.path.join(current_dir , model_path)


env = Spacecraft_Env(args)
env.reset()
model = SAC.load(model_path, env =env )

episodes = args.max_nu_ep
with tf.device('/GPU:0'):
    for ep in range (episodes):
        obs = env.reset()
        done = False
        steps = 0
        model = SAC.load(model_path, env =env )
        while not done:
            steps = steps + 1
            # print("Step {}".format(steps))
            action = model.predict(obs)
            obs, reward, done, info = env.step(action[0])
    
env.close()
print("All EPISODES DONE ! ")


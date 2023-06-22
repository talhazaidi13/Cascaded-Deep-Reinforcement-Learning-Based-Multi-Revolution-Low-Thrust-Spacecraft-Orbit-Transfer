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


tf.debugging.set_log_device_placement(True)
import argparse
from config import args

# Assign the variables to the parsed arguments
max_steps_one_ep = args.max_steps_one_ep
max_nu_ep = args.max_nu_ep
weights_save_steps = args.weights_save_steps
buffer_size = args.buffer_size
gamma = args.gamma
lr = args.lr

today = date.today()
day = today.strftime('%b-%d-%Y')
txt_dir = f'{int(time.time())}'
dir_name = day + "_" + txt_dir +'/'
current_dir = os.getcwd()
models_dir = os.path.join(current_dir , 'Model_training_weights', dir_name)
logdir = os.path.join(current_dir , 'Model_training_logs', dir_name)
 
if not os.path.exists(models_dir):
	os.makedirs(models_dir)
if not os.path.exists(logdir):
	os.makedirs(logdir)

env = Spacecraft_Env(args)
env.reset()

model = SAC('MlpPolicy', env, learning_rate=lr, buffer_size=buffer_size, gamma=gamma, tau=0.005, train_freq=1,  verbose=1, tensorboard_log=logdir)

n_steps = max_steps_one_ep*max_nu_ep
TIMESTEPS = weights_save_steps
with tf.device('/GPU:0'):
    for step in range (n_steps):
        print("Step {}".format(step + 1))
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=1, tb_log_name='SAC')
        file_name= str(step)+"_"+str(int(time.time()))
        model.save(f"{models_dir}/{file_name}")

print("All EPISODES DONE ! ")







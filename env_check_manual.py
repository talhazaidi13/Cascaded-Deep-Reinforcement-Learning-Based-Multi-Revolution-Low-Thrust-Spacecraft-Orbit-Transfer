from Spacecraft_Env import  Spacecraft_Env
env = Spacecraft_Env()
episodes = 50



for episode in range(episodes):
    done = False
    obs  = env. reset()
    print('obs', obs)
    print("env.observation_space", env.observation_space)
    print("env.observation_space.shape", env.observation_space.shape)
    print("env.action_space.shape", env.action_space.shape)
    print("env.action_space", env.action_space)

    while not done:
        random_action = env.action_space.sample()
       # print("action", random_action)
        obs, reward, done, info = env.step(random_action)
        #print('reward', reward)
        print('obs', obs, done)
 
 
 
    
'''
#################################################################

random_action = env.action_space.sample()
print("action", random_action)
from collections import deque
prev_actions = deque(maxlen = 30)
for i in range (30):
        random_action = env.action_space.sample()
        prev_actions.append(random_action)
print("prev_actions : ",prev_actions)
'''



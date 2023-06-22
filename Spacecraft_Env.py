####################################################################################################################################################################################################    
   # for continuos episode 
######################################################################################################################################################################################################    
  
import numpy as np
import gym
from gym import spaces
from enviornment import Enviornment 
##################################################################
import math
import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import matlab.engine         # IMPORTING MATLAB ENGINE
#import progressbar          
import gym
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.image as img
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
## enviornment start
import matlab.engine
import csv
#eng = matlab.engine.start_matlab();
from numpy.core.fromnumeric import shape         
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
import os.path
import time
from datetime import date
import matplotlib.cm as cm
import matplotlib.animation as animation
import tensorflow as tf 
from scenerios import cases
tf.debugging.set_log_device_placement(True)

LEN_GOAL = 30
##################################################################
class Spacecraft_Env(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a spacecraft env where agent learns to find the optimal satellite path. 
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  # metadata = {'render.modes': ['console']}
 
  def __init__(self, args):
    super(Spacecraft_Env, self).__init__()
    self.max_steps_one_ep = args.max_steps_one_ep
    chosen_case = cases[args.case]
    eng = matlab.engine.start_matlab();
    #env1 =  eng.Mat_env.resulting();
    self.env = Enviornment(eng, args)
    self.max_R_time = 500000000000
    self.F = chosen_case['force'][0]/1000 
    self.max_steps_in_ep = chosen_case['max_steps_in_ep'][0]
    
    number_of_revolutions = 200*50000
    self.segment_interval = self.env.segment  # in degrees
    self.seg1= self.segment_interval      # * np.pi/180      # in radians
    self.segment = (360/ self.segment_interval)
    max_possible_steps_in_episodes = number_of_revolutions * self.segment
    self.phi_normalizing_factor = max_possible_steps_in_episodes * self.seg1
    self.max_steps = number_of_revolutions *  self.segment 
    
    
    # define arrays for info
    self.h_history = [[]]
    self.hx_history = [[]] 
    self.hy_history = [[]] 
    self.ex_history = [[]] 
    self.ey_history = [[]] 
    self.phi_history = [[]]
    self.mass_history = [[]]
    self.alpha_history = [[]]
    self.beta_history = [[]] 
    self.thrust_history = [[]]
  
     
    self.ecc_history = [[]]
    self.a_history = [[]]
    self.inclination_history = [[]]
    self.ecc_history_nd = [[]]
    self.a_history_nd = [[]]
    self.inclination_history_nd = [[]]
    

    self.score_data = []
    self.score_detailed_data = [[]]

    
    ##
    self.seg_count = 0
    self.average = 0
    self.a_sum = 0
    self.time_rev = 0
    self.ep_time = 0
    self.ep_time_1 = 0
    self.T_rev = [[]]
    self.T_ep_success = []
    self.T_ep_success.append([])
    ##
    self.ep_counter = -1
    self.counter = 0
    self.MAIN_Episode = 1
    self.done_counter = 0
    self.ep_counter_SAC_step = -1
    self.ep_Cont_len_counter = 0
    self.success_counter = 0
    self.time_mat = 0
    self.acc_reward = 0

    current_dir = os.getcwd()
    today = date.today()
    self.day = today.strftime('%b-%d-%Y')
    self.txt_dir = f"{int(time.time())}"
    self.dir_name = self.day + "_" + self.txt_dir
    folder_path = os.path.join(current_dir,'plots', self.dir_name)
    folders = ['h', 'hx', 'hy', 'ex', 'ey', 'a', 'ecc', 'inc', 'mass', 'phi', 'sum_reward', 'time', 'Successful_episodes']
    folder_paths = [os.path.join(f"{folder_path}/{folder}") for folder in folders]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        for folder_path_1 in folder_paths:
            os.makedirs(folder_path_1)
    
   
    self.completeName_successful = os.path.join(folder_path+"/output.dat")
    self.write_final_state = os.path.join(folder_path+"/final_state.dat")
    self.completeName_all_data= os.path.join(folder_path+"/output_all_data.dat")
    self.figure_file_ecc = os.path.join(folder_path + "/_ecc_history")
    self.figure_file_a = os.path.join(folder_path + "/_a_history")
    self.figure_file_inclination = os.path.join(folder_path + "/_inclination_history")   
    self.figure_file_reward = os.path.join(folder_path + "/_reward_history")
    self.figure_file_time_success = os.path.join(folder_path + "/_success_ep_time")
    self.figure_file_time_success_actual = os.path.join(folder_path, "_success_ep_time_actual")
    self.folder_path_h = os.path.join(f"{folder_paths[0]}/{'h'}")
    self.folder_path_hx = os.path.join(f"{folder_paths[1]}/{'hx'}")
    self.folder_path_hy = os.path.join(f"{folder_paths[2]}/{'hy'}")
    self.folder_path_ex = os.path.join(f"{folder_paths[3]}/{'ex'}")
    self.folder_path_ey = os.path.join(f"{folder_paths[4]}/{'ey'}")
    self.folder_path_a = os.path.join(f"{folder_paths[5]}/{'a'}")
    self.folder_path_ecc = os.path.join(f"{folder_paths[6]}/{'ecc'}")
    self.folder_path_inc = os.path.join(f"{folder_paths[7]}/{'inc'}")
    self.folder_path_mass = os.path.join(f"{folder_paths[8]}/{'mass'}")
    self.folder_path_phi = os.path.join(f"{folder_paths[9]}/{'phi'}")
    self.folder_path_sum_reward = os.path.join(f"{folder_paths[10]}/{'reward'}") 
    self.folder_path_time = os.path.join(f"{folder_paths[11]}/{'ep_time'}")
    self.successful_episodes = os.path.join(f"{folder_paths[12]}")
   
   
    
    self.action_space = spaces.Box(low=np.float32([-np.pi, -np.pi/2]),
                                       high=np.float32([np.pi, np.pi/2]), shape=(2,),
                                        dtype=np.float32)

    self.observation_space = spaces.Box(low=-15, high=15,
                                        shape=(7,), dtype=np.float64)


  
  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    self.state = np.array(self.env.reset_csv())
    self.ND_state = np.array(self.env.DimtoNonDim_states(self.state, self.max_R_time, 2*np.pi))
    self.observation_notime = np.delete(self.ND_state, 6)
    
    self.ep_Cont_len_counter = self.ep_fixed_len_counter = 0
    self.seg_count = 0
    self.ep_time = self.ep_time_1 = 0
    self.ep_counter += 1
    self.ep_counter_SAC_step += 1
    self.acc_reward = 0
    self.seg_change_indx = []

    ecc = math.sqrt(self.state[3]**2 + self.state[4]**2)
    self.ecc_history.append([ecc])
    self.a_history.append([(self.state[0]**2 / 398600.4418) / (1 - ecc**2)])
    self.inclination_history.append([(math.asin(math.sqrt(self.state[1]**2 + self.state[2]**2) / self.state[0]) / np.pi) * 180])
    
    ecc_ND = math.sqrt(self.ND_state[3]**2 + self.ND_state[4]**2)
    self.ecc_history_nd.append([ecc_ND])
    self.a_history_nd.append([(self.ND_state[0]**2 / 398600.4418) / (1 - ecc_ND**2)])
    self.inclination_history_nd.append([(((math.asin(math.sqrt((self.ND_state[1]**2)+(self.ND_state[2]**2))/self.ND_state[0])) / np.pi)*180)/10] )
    
    self.h_history.append([self.state[0]])
    self.hx_history.append([self.state[1]])
    self.hy_history.append([self.state[2]])
    self.ex_history.append([self.state[3]])
    self.ey_history.append([self.state[4]])
    self.phi_history.append([self.state[5]])
    self.mass_history.append([self.state[-1]])
    self.alpha_history.append([])
    self.beta_history.append([])
    self.thrust_history.append([]) 
    self.score_data.append([])
    self.score_detailed_data.append([])
    
    self.T_rev.append([])
    self.time_mat  = 0
    
    self.observation = self.observation_notime
 
    return self.observation  

  
  def step(self, action):
    action = np.array([action[0], action[1] ,self.F])    #0.34fixed thrust  0.089 gives error near GEO
    ## Convert Non dimensional values to dimensional to p#ass through the matlab step function
    self.observation_Notime = [self.observation[0], self.observation[1], self.observation[2], self.observation[3], self.observation[4], self.observation[5], 0, self.observation[6]]
    self.D_state = np.array(self.observation_Notime)
    self.D_state = self.env.NonDimtoDim_states(self.D_state , self.max_R_time ,  (2*np.pi))
    self.state1= np.array([self.D_state[0], self.D_state[1], self.D_state[2], self.D_state[3], self.D_state[4], self.D_state[5], self.D_state[6] ,self.D_state[7] ]).reshape((1,8))

    ## matlab step function
    mass = self.D_state[-1]
    self.state1[0][-1] = self.state1[0][-1] * -1          # making mass value negative before sending in matlab function env
    self.state1[0][6] = self.time_mat
    finalmass= self.env.m0 - np.abs(mass)
 
    self.next_state, self.reward, self.distance, self.dist_aei, terminal_flag, Remaining_time, target_state_parameters, redflag, time_in_days, ecc_nd, a_nd, i_nd,  eclipse_flag, segment_flag, time_bef_seg_change, time_after_seg_change,segment = self.env.step(self.state1, action[0], action[1], action[2], self.seg1, finalmass , self.env.I_sp, self.max_R_time, (2*np.pi) ,self.ep_Cont_len_counter)  
    self.next_state[-1] = self.next_state[-1] * -1      # making mass value positive for our networks
    self.time_mat = time_in_days * (60*60*24)
   
    eclipse_info = 0
    if eclipse_flag:
        action[2] = 0.0
        eclipse_info = 1
     
    self.next_state = np.array(self.next_state)
    self.ND_next_state = np.array(self.env.DimtoNonDim_states (self.next_state , self.max_R_time , (2*np.pi)))
    self.observation_next_state_Notime = [self.ND_next_state[0], self.ND_next_state[1], self.ND_next_state[2], self.ND_next_state[3], self.ND_next_state[4], self.ND_next_state[5], self.ND_next_state[7] ]
    self.observation = self.observation_next_state_Notime
    ## termianl flag
    self.done = terminal_flag
    # for plotting append data
    self.h_history[self.ep_counter-1].append(self.state1[0][0])
    self.hx_history[self.ep_counter-1].append(self.state1[0][1])
    self.hy_history[self.ep_counter-1].append(self.state1[0][2])
    self.ex_history[self.ep_counter-1].append(self.state1[0][3])
    self.ey_history[self.ep_counter-1].append(self.state1[0][4])
    self.phi_history[self.ep_counter-1].append(self.state1[0][5])
    self.ecc_history[self.ep_counter-1].append(target_state_parameters[0])
    self.a_history[self.ep_counter-1].append(target_state_parameters[1])
    self.inclination_history[self.ep_counter-1].append(target_state_parameters[2])
    self.ecc_history_nd[self.ep_counter-1].append(ecc_nd)
    self.a_history_nd[self.ep_counter-1].append(a_nd)
    self.inclination_history_nd[self.ep_counter-1].append(i_nd)
    self.mass_history[self.ep_counter-1].append(finalmass)
    self.alpha_history[self.ep_counter-1].append([action[0]])
    self.beta_history[self.ep_counter-1].append([action[1]])
    self.thrust_history[self.ep_counter-1].append([action[2]]) 
    self.acc_reward = self.acc_reward + self.reward[0]
    self.score_data[self.ep_counter-1] = self.acc_reward 
    self.score_detailed_data[self.ep_counter-1].append(self.acc_reward)
    
    # Increase counter value
    self.ep_Cont_len_counter = self.ep_Cont_len_counter + 1
    self.ep_fixed_len_counter = self.ep_fixed_len_counter + 1
    self.counter = self.counter + 1
 
    #### Time from matlab file
    if self.done:
      self.T_rev[self.ep_counter-1].append(time_in_days)   #matlab
    ##############################
    if self.ep_Cont_len_counter == 10000 and (self.ep_counter-1 ) % 10 == 0 and (self.ep_counter-1 ) != 0:
      self.env.plot_variable("Score_value", self.score_data, self.figure_file_reward, self.ep_counter, all_episode_plot_flag=1)
      
    if ((redflag==1) or (self.ep_Cont_len_counter % 5000 == 0)):
      self.env.plot_variable("H", self.h_history, self.folder_path_h, self.ep_counter)
      self.env.plot_variable("Hx", self.hx_history, self.folder_path_hx, self.ep_counter)
      self.env.plot_variable("Hy", self.hy_history, self.folder_path_hy, self.ep_counter)
      self.env.plot_variable("ex", self.ex_history, self.folder_path_ex, self.ep_counter)
      self.env.plot_variable("ey", self.ey_history, self.folder_path_ey, self.ep_counter)
      self.env.plot_variable("ecc", self.ecc_history, self.folder_path_ecc, self.ep_counter, flag_ter_values=2, tsp=target_state_parameters, tsp_indexes=[3])
      self.env.plot_variable("a", self.a_history, self.folder_path_a, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[4,5])
      self.env.plot_variable("inc", self.inclination_history, self.folder_path_inc, self.ep_counter, flag_ter_values=2, tsp=target_state_parameters, tsp_indexes=[6])
      self.env.plot_variable("mass", self.mass_history, self.folder_path_mass, self.ep_counter)
      self.env.plot_variable("Reward", self.score_detailed_data, self.folder_path_sum_reward, self.ep_counter)
      self.env.plot_variable("Phi", self.phi_history, self.folder_path_phi, self.ep_counter)
     
    if self.done:
      self.success_counter = self.success_counter +1
      self.T_ep_success.append([])
      self.T_ep_success[-2] = self.T_rev[self.ep_counter-1][-1]
      average = 0
      done_data = [(self.ep_counter-1),  self.ep_Cont_len_counter, self.acc_reward ,self.T_ep_success[-2], self.completeName_successful ]     
      self.env.writing_Successful_episodes ( int(self.success_counter), int(done_data[0]), int(done_data[1]), 
                                            (float(done_data[2])),  (float(done_data[3])),
                                            float(42164- (self.a_history[self.ep_counter-1][-1])), float(self.inclination_history[self.ep_counter-1][-1]),
                                            float(self.ecc_history[self.ep_counter-1][-1]),   
                                            float(self.h_history[self.ep_counter-1][-1]), float(self.hx_history[self.ep_counter-1][-1]), float(self.hy_history[self.ep_counter-1][-1]), 
                                            float(self.ex_history[self.ep_counter-1][-1]), float(self.ey_history[self.ep_counter-1][-1]), 
                                            self.completeName_successful  )
      self.env.writing_final_states(  float(self.h_history[self.ep_counter-1][-1]), float(self.hx_history[self.ep_counter-1][-1]), float(self.hy_history[self.ep_counter-1][-1]), 
                                            float(self.ex_history[self.ep_counter-1][-1]), float(self.ey_history[self.ep_counter-1][-1]), self.phi_history[self.ep_counter-1][-1],
                                            float(self.time_mat) ,  float(self.next_state[-1]*-1) ,self.write_final_state )
   
      ##############################################################
      self.env.plot_variable("ecc", self.ecc_history, self.figure_file_ecc, self.ep_counter, flag_ter_values=2, tsp=target_state_parameters, tsp_indexes=[3])
      self.env.plot_variable("a", self.a_history, self.figure_file_a, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[4,5])
      self.env.plot_variable("inc", self.inclination_history, self.figure_file_inclination, self.ep_counter, flag_ter_values=2, tsp=target_state_parameters, tsp_indexes=[6])
      self.env.plot_variable("Score_value", self.score_data, self.figure_file_reward, self.ep_counter, all_episode_plot_flag=1)
      
      
      if not os.path.exists(self.successful_episodes ):
          os.makedirs(self.successful_episodes)
      self.Successful_Episode_num = os.path.join(self.successful_episodes +"/_ep_"+ str(self.ep_counter-1) )
      if not os.path.exists(self.Successful_Episode_num ):
          os.makedirs(self.Successful_Episode_num )
          
      self.env.plot_variable("H", self.h_history, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_two_variable("hx_hy","hx","hy", self.hx_history,self.hy_history,  self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_two_variable("ex_ey","ex","ey", self.ex_history,self.ey_history,  self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("ecc", self.ecc_history, self.Successful_Episode_num, self.ep_counter, flag_ter_values=2, tsp=target_state_parameters, tsp_indexes=[3], flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("a(km)", self.a_history, self.Successful_Episode_num, self.ep_counter, flag_ter_values=1, tsp=target_state_parameters, tsp_indexes=[4,5], flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("inc(deg)", self.inclination_history, self.Successful_Episode_num, self.ep_counter, flag_ter_values=2, tsp=target_state_parameters, tsp_indexes=[6], flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("mass(kg)", self.mass_history, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("Reward", self.score_detailed_data, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("Phi(rad)", self.phi_history, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_two_variable("actions","alpha(rad)","beta(rad)", self.alpha_history,self.beta_history,  self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("Thrust(KN)", self.thrust_history, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      self.env.plot_variable("Time(KN)", self.T_rev, self.Successful_Episode_num, self.ep_counter, flag_saving_with_no_ep_nu =1)
      
 
    
     # Optionally we can pass additional info, we are not using that for now
    info = {}
    
    
    
    print("Ep:", self.ep_counter_SAC_step , "  Ep_len :", self.ep_fixed_len_counter , "  step_reward:" , self.reward[0] ,   "  Acc_reward: ", self.acc_reward,
          "  ecc:" , float(round(target_state_parameters[0],6)), "  a  :" , float(round(target_state_parameters[1],4)), "  inc:" , float(round(target_state_parameters[2],4)),
          "  a_diff:" , float(round(42164- (self.a_history[self.ep_counter-1][-1]),4)), "  time :", time_in_days, "  Seg :", segment*(180/3.14) ," eclipse_info : ", eclipse_info)
 
    
    self.env.writing_all_episodes_data ( int(self.ep_counter_SAC_step) , int(self.ep_fixed_len_counter), 
                                        float(round(self.state1[0][0],6)), float(round(self.state1[0][1],6)),float(round(self.state1[0][2],6)),float(round(self.state1[0][3],6)),
                                        float(round(self.state1[0][4],6)),float(round(self.state1[0][5],6)), 
                                        float(round(self.observation_Notime[0],6)),float(round(self.observation_Notime[1],6)),float(round(self.observation_Notime[2],6)),float(round(self.observation_Notime[3],6)),
                                        float(round(self.observation_Notime[4],6)),float(round(self.observation_Notime[5],6)), 
                                        float(round(target_state_parameters[0],6)), float(round(target_state_parameters[1],6)), float(round(target_state_parameters[2],6)), 
                                        float(round(action[0],6)), float(round(action[1],4)),float(round(action[2],4)), 
                                        float(round(self.reward[0],4)), float(round(self.acc_reward,4)), float(finalmass),
                                        target_state_parameters[7] , target_state_parameters[8], target_state_parameters[9], segment*(180/3.14), 
                                        self.completeName_all_data  )
     
    if (redflag) or (self.ep_fixed_len_counter > self.max_steps_in_ep ):    #50000 with 10 degree , 1000000 with 1 deg
      self.done=1
        
    
    return np.array(self.observation), self.reward[0], self.done, info
  

  def render(self):
    pass

  def close(self):
    pass
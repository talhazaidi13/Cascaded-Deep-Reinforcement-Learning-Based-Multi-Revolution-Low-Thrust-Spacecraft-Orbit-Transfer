import math
from operator import mod
from turtle import distance
import numpy as np
#import progressbar          
import gym
import random
import matplotlib.pyplot as plt
## enviornment start
import matlab.engine
import csv
#eng = matlab.engine.start_matlab();

from numpy.core.fromnumeric import shape         

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

import os.path
import re
import tensorflow as tf 
tf.debugging.set_log_device_placement(True)
from config import args
from scenerios import cases


#env =  eng.Triangle_1.resulting();
#state = list(env._data)
#print('states [h hx hy ex ey phi m^] : ', state)

#state_space= shape(state)
#print('State_Space : ',state_space[0])     

#alpha = 1.5;
#beta = 3;
#F = 2*0.001035;
#segment = (10*3.14)/180


class Enviornment:
    def __init__(self, eng, args):
        self.eng = eng
        
        chosen_case = cases[args.case]
        self.initial_state =  np.array((chosen_case['GTO_state']))  # [h;hx;hy;ex;ey;phi;time;fuel_burnt]   Initial GTO orbit state  
               
        ################################################################################################
        ######## following initialization for dimenless quantities required for Neural network  ########
        self.DU =  42164;                      #  distance unit, Km
        self.TU = math.sqrt((42164**3)/398600);     #  time unit, s   572.92
        self.SU = self.DU/self.TU;          #  speed unit, Km/sec
        self.MU = 398600.4418;                    #  mass Unit, Kg
        self.HU = self.DU * self.SU;  
        self.g0 = 9.81; # m/sec^2
       
        ################################################################################################
        # define path for dat files
        current_dir = os.getcwd()
        save_path_1 = os.path.join(current_dir, "csv_files")
        self.completeName_csvlist = os.path.join(save_path_1, "csvlist.dat") 
        self.completeName_csvlistinitialize = os.path.join(save_path_1, "csvlistinitialize.dat") 
        self.completeName_Rtimeinput = os.path.join(save_path_1, "Rtimeinput.dat")
        ################################################################################################ 
        self.mu         = 398600.4418        # global mu matlab
        
        self.segment    = 10*(3.14/180)      # 10 degree gap convert in radians
        self.I_sp       = chosen_case['Isp'][0]        # in sec
        self.m0         = chosen_case['m0'][0]           # kg
        self.F          = chosen_case['force'][0]/1000  
        self.tol_inc    = chosen_case['tol_inc'][0]
        self.tol_ecc    = chosen_case['tol_ecc'][0]
        self.tol_a      = chosen_case['tol_a'][0]
        self.m_dot = -self.F/self.I_sp/self.g0; # in kg/sec
         
        self.shadow_flag      = chosen_case['sh_flag'][0]
        self.transfer_case = args.case
        
        self.target_inc, self.target_ecc, self.target_a = 0, 0, 42164
        self.tol_a_low, self.tol_a_high = self.target_a - self.tol_a, self.target_a + self.tol_a
        
        w1 =    {   "a": chosen_case['w1_a'][0],       "e": chosen_case['w1_e'][0],      "i": chosen_case['w1_i'][0]   }
        w1_ =   {   "a_": chosen_case['w1_a_'][0],   "e_": chosen_case['w1_e_'][0],    "i_": chosen_case['w1_i_'][0]   }
        c1 =    {   "a": chosen_case['c1_a'][0],       "e": chosen_case['c1_e'][0],     "i": chosen_case['c1_i'][0]   }
        
        self.weights = {
            "w1": w1,
            "w1_": w1_,
            "c1": c1,
            "tau": chosen_case['tau'][0]
        }
        
        self.segment_flag = 0
        self.time_before_seg_change = []
        self.time_after_seg_change = 0
        self.time_before_seg_change_1 = 0
   
    def is_terminal(self,state_5):
        h, hx, hy, ex, ey= state_5
        mu = self.mu # global mu matlab
    
        ecc = math.sqrt((ex**2)+(ey**2))
        flag_ecc = 1 if  ecc < (self.tol_ecc) else 0   
        p = (h*h)/mu
        a = p/(1-ecc**2)
        flag_a = 1 if (self.tol_a_low) < a and a < (self.tol_a_high) else 0 
        i = ((math.asin(math.sqrt((hx**2)+(hy**2))/h)) / np.pi)*180
        flag_inc = 1 if i<(self.tol_inc)  else 0   
        flag = 1 if flag_inc and flag_ecc and flag_a else 0
            
        p_init = ((self.initial_state[0]**2))/mu;
        tol_error_a =  self.tol_a_high + (self.tol_a_high* (5/100))      # 5 percent of final target a  (SHOULD BE 2 PERCENT)
        initial_state_ecc = math.sqrt((self.initial_state[3]**2)+(self.initial_state[4]**2))
        a_init = p_init/(1-initial_state_ecc**2);
        
        if int(self.transfer_case) > 6:
            flag_for_error_a = 1 if (a < 35000) or  (a > 52500) else 0           ## only for super GTO case
        else:
            flag_for_error_a = 1 if (a > tol_error_a) or (a<22000) else 0        ## only for all other GTO cases
        red_flag = 1 if flag_for_error_a else 0
        monitor_a_flag = 1 if (a < a_init) or  (a > self.tol_a_high) else 0
             
        return flag, ecc, self.tol_ecc, a, self.tol_a_low, self.tol_a_high, i,self.tol_inc  , 0, flag_for_error_a, 0,monitor_a_flag, red_flag
    
    
    def get_seg_value(self, state_5):    
        state_5_1 = state_5[0]
        h, hx, hy, ex, ey, _ , _, _= state_5_1
        mu = self.mu  # global mu matlab
        
        tol_inc = 0.1;                   # tolerance of inclination +- deg    0.1  0.01
        tol_ecc = 0.01;                  # 0.00001 tolerance of eccentricity +0  0.01  0.00001
        tol_a = 42164 * (5/100)          # 5,2  perent                   
        # For First segment change
        ecc = math.sqrt((ex**2)+(ey**2));
        flag_ecc_seg = 1 if  ecc < (tol_ecc) else 0    
        p = (h*h)/mu;
        a = p/(1-ecc**2);
        flag_a_seg = 1 if  (self.target_a - tol_a) < a    and   a < (self.target_a +  tol_a) else 0 
        i = ((math.asin(math.sqrt((hx**2)+(hy**2))/h)) / np.pi)*180;
        flag_inc_seg = 1 if i< (tol_inc)  else 0  
        self.segment = 1*(3.14/180) if flag_inc_seg and flag_ecc_seg and flag_a_seg else 10*(3.14/180) 
        # For second segment change
        tol_inc = 0.1;                  # tolerance of inclination +- deg    0.1  0.01
        tol_ecc = 0.01;                 # 0.00001 tolerance of eccentricity +0  0.01  0.00001
        tol_a_2 =  200              #  300, 30   
        ecc = math.sqrt((ex**2)+(ey**2));
        flag_ecc_seg_2 = 1 if  ecc < (tol_ecc) else 0    
        p = (h*h)/mu;
        a = p/(1-ecc**2);
        flag_a_seg_2 = 1 if  (self.target_a - tol_a) < a    and   a < (self.target_a +  tol_a) else 0 
        i = ((math.asin(math.sqrt((hx**2)+(hy**2))/h)) / np.pi)*180;
        flag_inc_seg_2 = 1 if i< (tol_inc)  else 0  
        if self.transfer_case in [2, 4, 6, 8]:        # for 2nd DRL case always it is 0.1 degree
            self.segment = 0.1*(3.14/180)
        else:
            if flag_inc_seg_2 and flag_ecc_seg_2 and flag_a_seg_2:
                self.segment = 0.1*(3.14/180) 
            elif flag_inc_seg and flag_ecc_seg and flag_a_seg:
                self.segment = 1*(3.14/180) 
            else:
                self.segment = 10*(3.14/180) 
        
        return self.segment
        
            
    def reset_csv(self):
        state     = self.initial_state
        self.temp = state
        self.temp = np.append(self.temp, 0.5)
        self.temp = np.append(self.temp, 0.5)
        self.temp = np.append(self.temp, self.F)
        self.temp = np.append(self.temp, self.segment)
        self.temp = np.append(self.temp, self.m0)
        self.temp = np.append(self.temp, self.I_sp)
        with open(self.completeName_csvlistinitialize , 'w') as csvfile: 
		       csvwriter = csv.writer(csvfile)
		       csvwriter.writerow(self.temp)
		       csvfile.close()
		       #print(a)    
        return state
        
    
    def step( self, state, alpha, beta, F, segment, m01, I_sp, Max_R_time , phi_normalizing_factor,   timestep):
        self.temp  = state
        segment    = self.get_seg_value(state)
        if args.case in ['2','4','6','8']:
            segment = 0.1*(3.14/180) 
        # for checking eclipse
        eclipse_flag =  self.eng.Mat_env.chkEclipse()
        if eclipse_flag * self.shadow_flag:
            F = 0

        self.temp = np.append(self.temp, alpha)
        self.temp = np.append(self.temp, beta)
        self.temp = np.append(self.temp, F)
        self.temp = np.append(self.temp, segment)
        self.temp = np.append(self.temp, m01)
        self.temp = np.append(self.temp, I_sp) 
        with open(self.completeName_csvlist, 'w') as csvfile: 
		       csvwriter = csv.writer(csvfile)
		       csvwriter.writerow(self.temp)
		       csvfile.close()
         
        a1 =  self.eng.Mat_env.resulting()
        #print("136env: state", a1)
        a3 = list(a1._data)
        t_state=a3[0:8]    # getting next state
        PropellentBurnt = t_state[7]
        time_in_days =  t_state[6] / (60*60*24)
        if (round(t_state[5], 2) % (round((2*np.pi), 2)) ) == 0:
            t_state[5] = 0
       
        ## Calculating Reward Function (passing ND values)
        new_state  = [t_state[0], t_state[1], t_state[2], t_state[3], t_state[4]]
        ND_state = self.DimtoNonDim_states (t_state , Max_R_time, phi_normalizing_factor)
        new_state_ND  = [ND_state[0], ND_state[1], ND_state[2], ND_state[3], ND_state[4]]
        prev_state = [state[0][0], state[0][1], state[0][2], state[0][3], state[0][4]]  
        ND_Previous_state = self.DimtoNonDim_states (state[0] , Max_R_time, phi_normalizing_factor)
        prev_state_ND  = [ND_Previous_state[0], ND_Previous_state[1], ND_Previous_state[2], ND_Previous_state[3], ND_Previous_state[4]]
        done,ecc, tol_ecc, a, tol_a_low, tol_a_high,  i,tol_inc, flag_ecc, flag_a, flag_inc, monitor_a_flag,redflag   =self.is_terminal(t_state[0:5])
        target_state_parameters = [ecc,a, i , tol_ecc, tol_a_low, tol_a_high, tol_inc,flag_ecc, flag_a, flag_inc  ]
        reward, distance, dist_aei, ecc_new, a_new, i_new, ecc_target, a_target, i_target = Enviornment.Reward(self, prev_state_ND,new_state_ND,   prev_state,new_state,   done,redflag,monitor_a_flag,   timestep)  #ex,ey,h,Remaining_time
          
        return t_state , reward, distance, dist_aei, 1==done , 1, target_state_parameters, redflag, time_in_days , ecc_new, a_new, i_new, eclipse_flag , self.segment_flag, self.time_before_seg_change_1, self.time_after_seg_change, segment
    
    
    
    def DimtoNonDim_states (self, state , Max_R_time, phi_normalizing_factor):
        h_1, hx_1, hy_1, ex_1, ey_1, phi_1, time_1, fuel_1 = state
        h_1 = h_1 / self.HU   # h
        hx_1 = hx_1 / self.HU   # hx
        hy_1  =  hy_1  / self.HU   # hy 
        phi_1 = phi_1 / phi_normalizing_factor
        time_1  = time_1  / Max_R_time  # accumulating time
        fuel_1 = fuel_1 / self.m0   # fuel burnt
        return [h_1, hx_1, hy_1, ex_1, ey_1, phi_1, time_1, fuel_1]
    
    def NonDimtoDim_states (self, state , Max_R_time, phi_normalizing_factor):
        h_1, hx_1, hy_1, ex_1, ey_1, phi_1, time_1, fuel_1 = state      
        h_1 = h_1 * self.HU   # h
        hx_1 = hx_1 * self.HU   # hx
        hy_1  =  hy_1  * self.HU   # hy 
        phi_1 = phi_1 * phi_normalizing_factor
        time_1  = time_1  * Max_R_time  # accumulating time
        fuel_1 = fuel_1 * self.m0   # fuel burnt
        return [h_1, hx_1, hy_1, ex_1, ey_1, phi_1, time_1, fuel_1]
    
    
    
    def Reward (self,  prev_state_ND,new_state_ND,   prev_state,new_state,  done,redflag,monitor_a_flag,   timestep):
       w1_aei  = [self.weights["w1"]["a"] , self.weights["w1"]["e"], self.weights["w1"]["i"]]                            
       w1_aei_ = [self.weights["w1_"]["a_"] , self.weights["w1_"]["e_"], self.weights["w1_"]["i_"] ]       
       c1_aei  = [self.weights["c1"]["a"] , self.weights["c1"]["e"], self.weights["c1"]["i"]]   
       tauu_aei= self.weights["tau"]

       starget = [ [1*(129640.2292/ self.HU)]  ,[0] , [0], [0], [0] ]
       
       ecc_prev = math.sqrt((prev_state_ND[3])**2 + (prev_state_ND[4])**2)
       ecc_new = math.sqrt((new_state_ND[3])**2 + (new_state_ND[4])**2)
       ecc_target = math.sqrt((starget[3][0])**2 + (starget[4][0])**2)
    
       a_prev = (((prev_state_ND[0])**2) /1) / ( 1- (ecc_prev **2))
       a_new = (((new_state_ND[0])**2) /1) / ( 1- (ecc_new **2))
       a_target = (((starget[0][0])**2) /1) / ( 1- (ecc_target **2))
       
       i_prev = ((math.asin (math.sqrt((prev_state_ND[1]**2)+(prev_state_ND[2]**2))/prev_state_ND[0])) / np.pi)*180
       i_new = ((math.asin (math.sqrt((new_state_ND[1]**2)+(new_state_ND[2]**2))/new_state_ND[0])) / np.pi)*180
       i_target = ((math.asin (math.sqrt((starget[1][0]**2)+(starget[2][0]**2))/starget[0][0])) / np.pi)*180
       
       i_prev = i_prev /10
       i_new = i_new /10
       i_target = i_target /10
    
       st_a_e_i_prev = [[a_prev], [ecc_prev], [i_prev]]
       st_a_e_i_new = [[a_new], [ecc_new], [i_new]]
       st_a_e_i_target = [[a_target], [ecc_target], [i_target]]
       
       exp_value_t_aei      =  0 
       exp_value_t_plus_1_aei =  0 
       
       for i in range(0,3):
          exp_value_t_aei        =  exp_value_t_aei        + ( c1_aei[i] * math.exp(-(w1_aei_[i] * abs(np.subtract(st_a_e_i_prev[i], st_a_e_i_target[i])) ) ) )
          exp_value_t_plus_1_aei =  exp_value_t_plus_1_aei + ( c1_aei[i] * math.exp(-(w1_aei_[i] * abs(np.subtract(st_a_e_i_new[i], st_a_e_i_target[i])) ) ) )
       phi_st_aei        = - np.dot( np.transpose(np.array(w1_aei)) , abs(np.subtract(st_a_e_i_prev ,st_a_e_i_target))) + exp_value_t_aei       -(0.03*monitor_a_flag)
       phi_st_plus_1_aei = - np.dot( np.transpose(np.array(w1_aei)) , abs(np.subtract(st_a_e_i_new ,st_a_e_i_target))) + exp_value_t_plus_1_aei -(0.03*monitor_a_flag)
       reward_t_aei = phi_st_plus_1_aei  - phi_st_aei - tauu_aei + (100 * done) - (5*redflag) -(0.00*monitor_a_flag)    
       
       diff_a  = math.sqrt(( st_a_e_i_target[0][0] - st_a_e_i_new[0][0])**2)
       diff_b = math.sqrt((st_a_e_i_target[1][0] - st_a_e_i_new[1][0])**2)
       diff_c = math.sqrt((st_a_e_i_target[2][0] - st_a_e_i_new[2][0])**2)
       dist_aei = diff_a + diff_b  + diff_c
       return [reward_t_aei[0]], 0, dist_aei, ecc_new, a_new, i_new,  ecc_target, a_target, i_target   
   
   
  
    
    def writing_Successful_episodes( self,success_ep_counter, episode, len_episode, score, time,a_last, Inc_last, ecc_last, h_last, hx_last,hy_last,ex_last,ey_last, completeName_successful  ):
        self.temp = ['Succ_ep_counter : ', success_ep_counter, '    ', '    ',
             'ep : ', episode, '    ', '    ', 'ep_length : ', len_episode, '    ', '    ',
             'score: ', score, '    ', '    ', 'time : ', time, '    ', '    ',
             'targ-a[-1]: ', a_last, '    ', '    ',
             'inc[-1]: ', Inc_last, '    ', '    ', 'ecc[-1]: ', ecc_last, '    ', '    ',
             'h[-1]: ', h_last, '    ', '    ', 'hx[-1]: ', hx_last, '    ', '    ',
             'hy[-1]: ', hy_last, '    ', '    ', 'ex[-1]: ', ex_last, '    ', '    ',
             'ey[-1]: ', ey_last]            
        with open(completeName_successful , 'a') as csvfile: 
		        csvwriter = csv.writer(csvfile)
		        csvwriter.writerow(self.temp)
		        csvfile.close()
		        #print(a)      
    
    def writing_all_episodes_data( self, episode, ep_fixed_len_counter, h,hx,hy,ex,ey,phi,ND_h,ND_hx,ND_hy,ND_ex,ND_ey,ND_phi,ecc,a,inc, alpha,beta,thrust, reward_step, score,mass, flag_ecc,flag_a,flag_inc,segment,completeName_successful  ):
        self.temp = ['ep : ', episode, '    ', 'ep_step : ', ep_fixed_len_counter, '    ', 'state : ', h, hx, hy, ex, ey, phi, '    ', 'ecc,a,inc:', ecc, a, inc, '    ', 'Reward,score : ', reward_step, score, '    ', '    ',
                     'Normalized_State: ', ND_h, ND_hx, ND_hy, ND_ex, ND_ey, ND_phi, '    ', '    ', 'action_values : ', alpha, beta,thrust, '    ', '    ', 'Mass : ', mass, '    ', '    ', 'flag_ecc : ', flag_ecc, '    ',
                     'flag_a : ', flag_a, '    ', 'flag_inc : ', flag_inc, '    ', 'Segment : ', segment]
        with open(completeName_successful , 'a') as csvfile: 
		        csvwriter = csv.writer(csvfile)
		        csvwriter.writerow(self.temp)
		        csvfile.close()
		        #print(a)
                
    def writing_final_states ( self, h,hx,hy,ex,ey,phi,time,mass, completeName_1 ):
        self.temp = []
        self.temp = np.append(self.temp, h)                                                           
        self.temp = np.append(self.temp, hx )                      
        self.temp = np.append(self.temp, hy)        
        self.temp = np.append(self.temp, ex)   
        self.temp = np.append(self.temp, ey)           #   [h;hx;hy;ex;ey;phi;time;fuel_burnt]  
        self.temp = np.append(self.temp, phi)   
        self.temp = np.append(self.temp, time)  
        self.temp = np.append(self.temp, mass) 
                  
        with open(completeName_1 , 'a') as csvfile: 
		        csvwriter = csv.writer(csvfile)
		        csvwriter.writerow(self.temp)
		        csvfile.close()
          
    
    def plot_variable(env, name, hist, folder_path, ep_counter, flag_ter_values= None,  tsp=None, tsp_indexes=None, all_episode_plot_flag=None, flag_saving_with_no_ep_nu =None):
        plt.figure(figsize=(12, 8))
        if all_episode_plot_flag == 1:
             plt.plot(hist[0:-2]) 
        else:
             plt.plot(hist[ep_counter-1])     
        plt.title(f'state parameters values {name}')
        plt.ylabel(f'{name.lower()} values')
        if flag_ter_values == 1:
            plt.axhline(tsp[tsp_indexes[0]], color='r', linestyle='-')
            plt.axhline(tsp[tsp_indexes[1]], color='r', linestyle='-')
        elif flag_ter_values == 2:  
            plt.axhline(tsp[tsp_indexes[0]], color='r', linestyle='-')
            
        plt.grid(True)
        
        if flag_saving_with_no_ep_nu  == 1:
            plt.savefig(folder_path+"/_"+name + ".png")
        else:
            plt.savefig(folder_path+"_ep_"+str(ep_counter-1) + ".png")
            
        plt.close()
    
    def plot_two_variable(env, name, a,b, hist_1, hist_2, folder_path, ep_counter, flag_ter_values= None,  tsp=None, tsp_indexes=None , flag_saving_with_no_ep_nu =None ):
        plt.figure(figsize=(12, 8))
        plt.plot(hist_1[ep_counter-1], c='b', label=a, linewidth=1.5)     
        plt.plot(hist_2[ep_counter-1], c='r', label=b, linewidth=1.5)
        plt.legend()
        plt.grid(True)
        plt.title(f'state parameters values {name}')
        plt.ylabel(f'{name.lower()} values')
        if flag_ter_values == 1:
            plt.axhline(tsp[tsp_indexes[0]], color='r', linestyle='-')
            plt.axhline(tsp[tsp_indexes[1]], color='r', linestyle='-')
        
        if flag_saving_with_no_ep_nu  == 1:
            plt.savefig(folder_path+"/_"+name + ".png")
        else:
            plt.savefig(folder_path+"_ep_"+str(ep_counter-1) + ".png")
            
        plt.close()
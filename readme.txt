

############################################__________GTO1- GEO_________##########################################################################
# ## First network weights training start from GEO are as follow.        Time: 116.75 days+ 0.25days  = 117days
model_path ="E:/RL_project_outputs/plots/finalweights/23197_1671114199_GTO1_1stnet.zip" #Dec-14-2022_1670999182/23197_1671114199.zip  for GTO a0.5 e0.1 i0.1 start GTO  end following comment line    
# ## GTO - GEO first endpoints   self.state= np.array([129639.305662,80.60863664,-51.31518440,-0.0001912370795,-0.00133784952,4.88,0,171.924])  #a0.5, e0.1 i0.1 a0.5 e0.01 i0.1
# ## Then with above endpoints, check the whole trajectory and found the point just after the last shadow the new end points from where second netwrok trained is as follow
# ## GTO - GEO first early trajectory endpoints: self.state= np.array([129618.36017,231.805808,-205.689326,-0.000674,0.000658,0.348889,0,173.45981452])   # ,ep_step : 6375, ecc:0.0008  a:5, inc:0.145,    time 116.75 days     mass 1029.172668937865 
##### Now trained the second network with above start point: in weights multiply a, e with 3. in def_ter i0.1, ecc 0.0002, a 0.6 in def_seg seg=0.1 always and values 0.1,0.1,a5km insecond 0.1, ecc0.01, a0.5 
# ## Second network weights training start from above endpoints are as follow
model_path = "E:/OneDrive - Kansas State University/Talha Zaidi Phd/Research/basics code/SAC_stable_baseline/models/Jan-26-2023_1674760417/4670_1674789424.zip"  # Time 0.31days inc<0.057 (exact 0.56), ecc<=0.00019, a<0.4km(exact0.34km)      
# Another second netwrok, starts from same above points and weights but with more better tolerance values and time, ie tol: ecc<0.00005 a<0.2 inc<0.08, time: 0.25  
model_path = "E:/RL_project_outputs/plots/finalweights/7830_1674817940_GTO1_2ndnet.zip"   # Hayat comp, weights path D:\NASA_project\models\Jan-26-2023_1674774560\7830_1674817940.zip
#####################################################################################################################################################


# #############################################__________SACpaper GTO- GEO_________##########################################################################
# ### ### Total_time = 145.19+1.24 = 146.43 days with matching or better toelrance values!
# ### ### First model, end conditions, a55_ecc0.01_inc0.1, getseg_0.1,0.1,5%..0.1,0.1,70
# ## First model, end states self.state = np.array([129705.88617642476,-17.661511149943497,82.29267866601921,0.008180683884052145,0.005754129819996143,0.4448333333333312,0,149.19048617296883]) #ecc 0.0099,a47,i0.03716
# model_path = "E:/RL_project_outputs/plots/finalweights/8117_1669700993_SAC_paper_1stnet.zip" # hayat pc "D:/NASA_project/models/Nov-28-2022_1669663038/8117_1669700993.zip"  # sac paper _ Time 145.19 days  is_ter 0.1 0.01 55  #def_seg 0.1 0.1 5%, 0.1 0.1 70, 
# ###### Now trained the second network with above start point: in weights multiply a*3, e*6 .in def_ter i0.1, ecc 0.0009 a36  def_seg seg=0.1 always ep_length_set=5000
model_path = "E:/OneDrive - Kansas State University/Talha Zaidi Phd/Research/basics code/SAC_stable_baseline/models/Feb-01-2023_1675311068_SAC_GTO_2ndnet/11128_1675372524.zip" #Time 1.24, ecc0.00089 i<0.07 a<11
# ######################################################################################################################################################



# #############################################__________GTO2- GEO_________###############################################################################
# ### First model, end conditions, a50_ecc0.1_inc0.1, getseg_0.1,0.1,5%..0.1,0.1,200   Time: 70.08 + 0.31 days = 70.39 days 
# ## First model, end states ,self.state = np.array([129639.13631,-217.707577,-234.640998,-0.000326,0.000801,1.221111,0,36.111])  # final chosen one ep_step : 3716, ecc:0.0008  a:5, inc:0.13,    time 70.0 days        
# model_path = "E:/RL_project_outputs/plots/finalweights/15440_1674943155_GTO2_1stnet.zip"  #witheclipse  69.1 days  =>70.2 (70.0) with just after eclipse ecc0.0008 a0.5(5in used-one state) i0.1(0.14)       #orignal_path hayatpc: "D:/NASA_project/models/Jan-27-2023_1674866586/15440_1674943155.zip"
# ###### Now trained the second network with above start point: in weights multiply a, e with 3. in def_ter i0.1, ecc 0.00009, a 0.1 in def_seg seg=0.1 always 
# ############# Second_Netwrok end conditions, a0.08km_ecc0.000087_inc0.06, getseg_0.1forall
# model_path = "E:/RL_project_outputs/plots/finalweights/11495_1675448339_GTO2_2ndnet.zip "   #time0.31days     #orignalpath hayatpc: D:/NASA_project/models/Feb-02-2023_1675389717/11495_1675448339.zip  
# ######################################################################################################################################################


#  ############################################__________superGTO- GEO_________###############################################################################
# ### First model, end conditions, a50_ecc0.1_inc0.1, getseg_0.1,0.1,5%..0.1,0.1,200   Time: 80.6 + 0.27 days = 80.87 days 
# ## First model, end states self.state = np.array([129624.62553,-160.07483,-86.112609,-0.000256813,0.00082013,3.97210,0,84.2832])   # final chosen ep:6, ecc:0.002(0.0003-0.0007)  a:10, inc:0.1,    time 80.6 days        
# model_path = "E:/OneDrive - Kansas State University/Talha Zaidi Phd/Research/basics code/SAC_stable_baseline/models/Feb-05-2023_1675579558_SuperGTO_1stnet/49128_1675873753.zip"   #a<10  ecc<0.002 (0.0003-0.0007) time 80.6-80.8
# ###### Now trained the second network with above start point: in weights multiply a*4, e*4.5 i*1.5. in def_ter i0.1, ecc 0.0002, a 0.1 in def_seg seg=0.1 always and SHADOW ON WHILE TRAINING 
# ############# Second_Netwrok end conditions, a0.5km_ecc0.000069_inc0.1, getseg_0.1forall, and start_point_is self.state = np.array([129624.62553,-160.07483,-86.112609,-0.000256813,0.00082013,3.97210,0,84.2832])
model_path = "E:/OneDrive - Kansas State University/Talha Zaidi Phd/Research/basics code/SAC_stable_baseline/models/Feb-10-2023_1676078567_SuperGTO_2ndnet/4359_1676106250.zip"    #a0.5 ecc0.000069 i0.1 time 0.27   plots:E:\RL_project_outputs\plots\txtdata\Feb-12-2023_1676247819
# ######################################################################################################################################################

import argparse
# Create ArgumentParser object 
from scenerios import cases

parser = argparse.ArgumentParser()
# Define command-line arguments with default values and help messages



# parser.add_argument('--GTO_state', type=float, nargs=7, default=[101055.56404, 8993.12420, -44988.20968, 0.634234170, 0.142292050, 1000.0, 0], help='GTO1 orbit state for first DRL [h,hx,hy,ex,ey,mass,time]')
# parser.add_argument('--Isp', type=int, default=1800, help='Specific Impulse Isp')
# parser.add_argument('--m0', type=int, default=1200, help='Mass of spacecraft Kg')
# parser.add_argument('--force', type=int, default=0.311, help='Constant thrust magnitude in N')
# parser.add_argument('--tol_inc', type=float, default=0.1, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=0.1 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=0.5  , help='tolerance for semimajoraxis km')
# parser.add_argument('--w1_a', type=float, default=1000 , help='Reward function weights for w_a')
# parser.add_argument('--w1_e', type=float, default=2010, help='Reward function weights for w_ecc')
# parser.add_argument('--w1_i', type=float, default=300 , help='Reward function weights for w_inc')
# parser.add_argument('--w1_a_', type=float, default= 0.010 , help='Reward function weights for w_a_')
# parser.add_argument('--w1_e_', type=float, default=0.000000198, help='Reward function weights for w_ecc_')
# parser.add_argument('--w1_i_', type=float, default=0.0003 , help='Reward function weights for w_inc_')
# parser.add_argument('--c1_a', type=float, default=500 , help='Reward function weights for c_a')
# parser.add_argument('--c1_e', type=float, default=700, help='Reward function weights for c_ecc')
# parser.add_argument('--c1_i', type=float, default=300 , help='Reward function weights for c_inc')
# parser.add_argument('--tau', type=float, default=0.003 , help='Reward function constant tau value')



# parser.add_argument('--GTO_state', type=float, nargs=7, default=[101055.56404, 8993.12420, -44988.20968, 0.634234170, 0.142292050, 1000.0, 0], help='GTO1 orbit state for Second DRL [h,hx,hy,ex,ey,mass,time]')
# parser.add_argument('--Isp', type=int, default=1800, help='Specific Impulse Isp')
# parser.add_argument('--m0', type=int, default=1200, help='Mass of spacecraft Kg')
# parser.add_argument('--force', type=int, default=0.311, help='Constant thrust magnitude in N')
# parser.add_argument('--tol_inc', type=float, default=0.08, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=0.00005 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=0.2  , help='tolerance for semimajoraxis km')
# parser.add_argument('--w1_a', type=float, default=1000*3 , help='Reward function weights for w_a')
# parser.add_argument('--w1_e', type=float, default=700*3, help='Reward function weights for w_ecc')
# parser.add_argument('--w1_i', type=float, default=200 , help='Reward function weights for w_inc')
# parser.add_argument('--w1_a_', type=float, default= 0.010*3 , help='Reward function weights for w_a_')
# parser.add_argument('--w1_e_', type=float, default=0.00000198*3, help='Reward function weights for w_ecc_')
# parser.add_argument('--w1_i_', type=float, default=0.0003 , help='Reward function weights for w_inc_')
# parser.add_argument('--c1_a', type=float, default=500*3 , help='Reward function weights for c_a')
# parser.add_argument('--c1_e', type=float, default=700*3, help='Reward function weights for c_ecc')
# parser.add_argument('--c1_i', type=float, default=300 , help='Reward function weights for c_inc')
# parser.add_argument('--tau', type=float, default=0.003 , help='Reward function constant tau value')

# parser.add_argument('--GTO_state', type=float, nargs=7, default=[101055.56404, 8993.12420, -44988.20968, 0.634234170, 0.142292050, 1000.0, 0], help='GTO2 orbit state for first DRL[h,hx,hy,ex,ey,mass,time]')
# parser.add_argument('--Isp', type=int, default=3300, help='Specific Impulse Isp')
# parser.add_argument('--m0', type=int, default=450, help='Mass of spacecraft Kg')
# parser.add_argument('--force', type=int, default=0.2007, help='Constant thrust magnitude in N')
# parser.add_argument('--tol_inc', type=float, default=0.1, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=0.1 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=50  , help='tolerance for semimajoraxis km')

# parser.add_argument('--w1_a', type=float, default=1000 , help='Reward function weights for w_a')
# parser.add_argument('--w1_e', type=float, default=2010, help='Reward function weights for w_ecc')
# parser.add_argument('--w1_i', type=float, default=300 , help='Reward function weights for w_inc')
# parser.add_argument('--w1_a_', type=float, default= 0.010 , help='Reward function weights for w_a_')
# parser.add_argument('--w1_e_', type=float, default=0.000000198, help='Reward function weights for w_ecc_')
# parser.add_argument('--w1_i_', type=float, default=0.0003 , help='Reward function weights for w_inc_')
# parser.add_argument('--c1_a', type=float, default=500 , help='Reward function weights for c_a')
# parser.add_argument('--c1_e', type=float, default=700, help='Reward function weights for c_ecc')
# parser.add_argument('--c1_i', type=float, default=300 , help='Reward function weights for c_inc')
# parser.add_argument('--tau', type=float, default=0.003 , help='Reward function constant tau value')

# parser.add_argument('--GTO_state', type=float, nargs=7, default=[101055.56404, 8993.12420, -44988.20968, 0.634234170, 0.142292050, 1000.0, 0], help='GTO1 orbit state for Second DRL [h,hx,hy,ex,ey,mass,time]')
# parser.add_argument('--Isp', type=int, default=3300, help='Specific Impulse Isp')
# parser.add_argument('--m0', type=int, default=450, help='Mass of spacecraft Kg')
# parser.add_argument('--force', type=int, default=0.2007, help='Constant thrust magnitude in N')
# parser.add_argument('--tol_inc', type=float, default=0.07, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=00.000088 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=0.09  , help='tolerance for semimajoraxis km')
# parser.add_argument('--tol_inc', type=float, default=0.1, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=0.1 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=0.5  , help='tolerance for semimajoraxis km')
# parser.add_argument('--w1_a', type=float, default=500*3 , help='Reward function weights for w_a')
# parser.add_argument('--w1_e', type=float, default=700*3, help='Reward function weights for w_ecc')
# parser.add_argument('--w1_i', type=float, default=200 , help='Reward function weights for w_inc')
# parser.add_argument('--w1_a_', type=float, default= 0.010*3 , help='Reward function weights for w_a_')
# parser.add_argument('--w1_e_', type=float, default=0.00000198*3, help='Reward function weights for w_ecc_')
# parser.add_argument('--w1_i_', type=float, default=0.0003 , help='Reward function weights for w_inc_')
# parser.add_argument('--c1_a', type=float, default=50*3 , help='Reward function weights for c_a')
# parser.add_argument('--c1_e', type=float, default=30*3, help='Reward function weights for c_ecc')
# parser.add_argument('--c1_i', type=float, default=5 , help='Reward function weights for c_inc')
# parser.add_argument('--tau', type=float, default=0.003 , help='Reward function constant tau value')

# parser.add_argument('--GTO_state', type=float, nargs=7, default=[101055.56404, 8993.12420, -44988.20968, 0.634234170, 0.142292050, 1000.0, 0], help='GTO3 orbit state for first DRL [h,hx,hy,ex,ey,mass,time]')
# parser.add_argument('--Isp', type=int, default=3000, help='Specific Impulse Isp')
# parser.add_argument('--m0', type=int, default=2000, help='Mass of spacecraft Kg')
# parser.add_argument('--force', type=int, default=0.35, help='Constant thrust magnitude in N')
# parser.add_argument('--tol_inc', type=float, default=0.1, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=0.01 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=55  , help='tolerance for semimajoraxis km')
# parser.add_argument('--w1_a', type=float, default=1000 , help='Reward function weights for w_a')
# parser.add_argument('--w1_e', type=float, default=2010, help='Reward function weights for w_ecc')
# parser.add_argument('--w1_i', type=float, default=300 , help='Reward function weights for w_inc')
# parser.add_argument('--w1_a_', type=float, default= 0.010 , help='Reward function weights for w_a_')
# parser.add_argument('--w1_e_', type=float, default=0.000000198, help='Reward function weights for w_ecc_')
# parser.add_argument('--w1_i_', type=float, default=0.0003 , help='Reward function weights for w_inc_')
# parser.add_argument('--c1_a', type=float, default=500 , help='Reward function weights for c_a')
# parser.add_argument('--c1_e', type=float, default=700, help='Reward function weights for c_ecc')
# parser.add_argument('--c1_i', type=float, default=300 , help='Reward function weights for c_inc')
# parser.add_argument('--tau', type=float, default=0.003 , help='Reward function constant tau value')

# parser.add_argument('--GTO_state', type=float, nargs=7, default=[101055.56404, 8993.12420, -44988.20968, 0.634234170, 0.142292050, 1000.0, 0], help='GTO3 orbit state for Second DRL [h,hx,hy,ex,ey,mass,time]')
# parser.add_argument('--Isp', type=int, default=3000, help='Specific Impulse Isp')
# parser.add_argument('--m0', type=int, default=2000, help='Mass of spacecraft Kg')
# parser.add_argument('--force', type=int, default=0.35, help='Constant thrust magnitude in N')
# parser.add_argument('--tol_inc', type=float, default=0.1, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=0.0009 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=36  , help='tolerance for semimajoraxis km')
# parser.add_argument('--tol_inc', type=float, default=0.1, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=0.1 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=0.5  , help='tolerance for semimajoraxis km')
# parser.add_argument('--w1_a', type=float, default=500*3 , help='Reward function weights for w_a')
# parser.add_argument('--w1_e', type=float, default=700*6, help='Reward function weights for w_ecc')
# parser.add_argument('--w1_i', type=float, default=200 , help='Reward function weights for w_inc')
# parser.add_argument('--w1_a_', type=float, default= 0.010*3 , help='Reward function weights for w_a_')
# parser.add_argument('--w1_e_', type=float, default=0.00000198*6, help='Reward function weights for w_ecc_')
# parser.add_argument('--w1_i_', type=float, default=0.0003 , help='Reward function weights for w_inc_')
# parser.add_argument('--c1_a', type=float, default=50*3 , help='Reward function weights for c_a')
# parser.add_argument('--c1_e', type=float, default=30*6, help='Reward function weights for c_ecc')
# parser.add_argument('--c1_i', type=float, default=5 , help='Reward function weights for c_inc')
# parser.add_argument('--tau', type=float, default=0.003 , help='Reward function constant tau value')

# parser.add_argument('--GTO_state', type=float, nargs=7, default=[101055.56404, 8993.12420, -44988.20968, 0.634234170, 0.142292050, 1000.0, 0], help='Super-GTO orbit state for first DRL [h,hx,hy,ex,ey,mass,time]')
# parser.add_argument('--Isp', type=int, default=3300, help='Specific Impulse Isp')
# parser.add_argument('--m0', type=int, default=1200, help='Mass of spacecraft Kg')
# parser.add_argument('--force', type=int, default=0.4015, help='Constant thrust magnitude in N')
# parser.add_argument('--tol_inc', type=float, default=0.1, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=0.1 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=50  , help='tolerance for semimajoraxis km')
# parser.add_argument('--tol_inc', type=float, default=0.1, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=0.1 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=0.5  , help='tolerance for semimajoraxis km')
# parser.add_argument('--w1_a', type=float, default=1000 , help='Reward function weights for w_a')
# parser.add_argument('--w1_e', type=float, default=4010, help='Reward function weights for w_ecc')
# parser.add_argument('--w1_i', type=float, default=300 , help='Reward function weights for w_inc')
# parser.add_argument('--w1_a_', type=float, default= 0.010 , help='Reward function weights for w_a_')
# parser.add_argument('--w1_e_', type=float, default=0.000000000198, help='Reward function weights for w_ecc_')
# parser.add_argument('--w1_i_', type=float, default=0.0003 , help='Reward function weights for w_inc_')
# parser.add_argument('--c1_a', type=float, default=500 , help='Reward function weights for c_a')
# parser.add_argument('--c1_e', type=float, default=2000, help='Reward function weights for c_ecc')
# parser.add_argument('--c1_i', type=float, default=300 , help='Reward function weights for c_inc')
# parser.add_argument('--tau', type=float, default=0.003 , help='Reward function constant tau value')

# parser.add_argument('--GTO_state', type=float, nargs=7, default=[101055.56404, 8993.12420, -44988.20968, 0.634234170, 0.142292050, 1000.0, 0], help='Super-GTO orbit state for Second DRL [h,hx,hy,ex,ey,mass,time]')
# parser.add_argument('--Isp', type=int, default=3300, help='Specific Impulse Isp')
# parser.add_argument('--m0', type=int, default=1200, help='Mass of spacecraft Kg')
# parser.add_argument('--force', type=int, default=0.4015, help='Constant thrust magnitude in N')
# parser.add_argument('--tol_inc', type=float, default=0.1, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=0.000069 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=0.5  , help='tolerance for semimajoraxis km')
# parser.add_argument('--tol_inc', type=float, default=0.1, help='tolerance for inclination')
# parser.add_argument('--tol_ecc', type=float, default=0.1 , help='tolerance for eccentricity')
# parser.add_argument('--tol_a', type=float, default=0.5  , help='tolerance for semimajoraxis km')
# parser.add_argument('--w1_a', type=float, default=1000*4 , help='Reward function weights for w_a')
# parser.add_argument('--w1_e', type=float, default=4010*4.5, help='Reward function weights for w_ecc')
# parser.add_argument('--w1_i', type=float, default=300*1.5 , help='Reward function weights for w_inc')
# parser.add_argument('--w1_a_', type=float, default= 0.010*4 , help='Reward function weights for w_a_')
# parser.add_argument('--w1_e_', type=float, default=0.000000000198*4.5, help='Reward function weights for w_ecc_')
# parser.add_argument('--w1_i_', type=float, default=0.0003*1.5 , help='Reward function weights for w_inc_')
# parser.add_argument('--c1_a', type=float, default=500*4 , help='Reward function weights for c_a')
# parser.add_argument('--c1_e', type=float, default=2000*4.5, help='Reward function weights for c_ecc')
# parser.add_argument('--c1_i', type=float, default=300*1.5 , help='Reward function weights for c_inc')
# parser.add_argument('--tau', type=float, default=0.003 , help='Reward function constant tau value')

parser.add_argument('--case', choices=cases.keys(), default='8', help='Choose one of the predefined cases, 1: GTO1_1st , 2: GTO1_2nd, 3: GTO2_1st , 4: GTO2_2nd, 5: GTO3_1st , 6: GTO3_2nd, 7: superGTO_1st , 8: superGTO_2nd,     ')

for arg_name, arg_vals in cases['1'].items():
    parser.add_argument(f'--{arg_name}_1', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case1')
    
for arg_name, arg_vals in cases['2'].items():
    parser.add_argument(f'--{arg_name}_2', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case2')
    
for arg_name, arg_vals in cases['3'].items():
    parser.add_argument(f'--{arg_name}_3', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case3')
    
for arg_name, arg_vals in cases['4'].items():
    parser.add_argument(f'--{arg_name}_4', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case4')
    
for arg_name, arg_vals in cases['5'].items():
    parser.add_argument(f'--{arg_name}_5', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case5')
    
for arg_name, arg_vals in cases['6'].items():
    parser.add_argument(f'--{arg_name}_6', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case6')
    
for arg_name, arg_vals in cases['7'].items():
    parser.add_argument(f'--{arg_name}_7', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case7')
    
for arg_name, arg_vals in cases['8'].items():
    parser.add_argument(f'--{arg_name}_8', type=type(arg_vals[0]), nargs=len(arg_vals), default=arg_vals, help=f'{arg_name} for case8')

parser.add_argument('--max_steps_one_ep', type=int, default=20000, help='Max number of steps in one episode ')
parser.add_argument('--max_nu_ep', type=int, default=600, help='Max number of episodes')
parser.add_argument('--weights_save_steps', type=int, default=500, help='Number of steps after which weights will save')
parser.add_argument('--buffer_size', type=int, default=10000, help='size of replay buffer')
parser.add_argument('--gamma', type=float, default=0.99, help='value of discounting parametr gamma')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')



# Parse the arguments
args = parser.parse_args()
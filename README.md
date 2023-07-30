# Cascaded-Deep-Reinforcement-Learning (CDRL) Based Multi-Revolution Low-Thrust-Spacecraft Orbit Transfer
We provide the code repository for our paper This repository includes the necessary code to replicate our experiments and utilize our DRL model for spacecraft trajectory planning. By accessing the repository, researchers and practitioners can benefit from our approach to efficiently transfer spacecraft to GEO using low-thrust propulsion systems.
CDRL based GTO to GEO transfer  | CDRL based Super-GTO to GEO transfer
:-: | :-:
<![CDRL based GTO to GEO transfer](/paper-outputs/GTO-GEO.gif) > | <![CDRL based GTO to GEO transfer](/paper-outputs/SuperGTO-GEO.gif) >
![CDRL based GTO to GEO transfer](/paper-outputs/a-e.PNG) 
![CDRL based GTO to GEO transfer](/paper-outputs/Orbital_Rotation_2.png)

## Files Description

- `config.py`: Contains the configurations or initial parameters to run the code.

- `Scenarios.py`: Contains the parameters for six transfer cases, which are as follows:
    - GTO-1 to GEO_1st network
    - GTO-1 to GEO_2nd network
    - GTO-2 to GEO_1st network
    - GTO-2 to GEO_2nd network
    - Super-GTO to GEO_1st network
    - Super-GTO to GEO_2nd network

- `Spacecraft_env.py`: Contains the gym structured environment, which includes `env.reset()`, `env.step()`, and `env.render()` functions.

- `environment.py`: Contains custom environment functions. These functions in `environment.py` are called by the gym environment in `Spacecraft_env.py`.

- `spacecraftEnivironment.m`: A MATLAB code used to calculate `env.step()` values.

- `environment.yml`: Contains all the required commands to recreate the Conda environment in any local system. This will recreate the exact same environment in which we trained our algorithms.

- `environment_info.txt`: Contains the versions of all installed packages present in our Conda environment.

- `test.py`: Python file which can be used to run the scenerios from pre trained weights.
- `test.sh`: Shell file that contains the code to run test.py file. You can select the case number and max number of episode and all other parameters which are defined in config file in here. <br>
             e.g If you want to run Case 1 i.e 'GTO-1 to GEO_1st network' then in test.sh file you will write as follow:
  ```python
  python test.py  --case 1  --max_nu_ep 100
  ```

- `train.py`: Python file which can be used to train the scenarios from scratch.
- `train.sh`: Shell file that contains the code to run train.py file. You can select the case number and max number of episode and all other parameters which are defined in config file in here. <br>
             e.g If you want to run Case 1 i.e 'GTO-1 to GEO_1st network' then in train.sh file you will write as follow:
  ```python
  python train.py  --case 1 --sh_flag 0
  ```
   Note:  Make sure that while training --sh_flag 0 
- `Final weights` folder: Contains the final trained weights for all six scenarios.
- `CSV Files` folder:  Contains Csv files which is used to communicate between Matlab and python programs data.
- `Plots` : The resulting plots from training or testing the DRL agents will be saved in plots folder.
- `Model_training_weights`: The resulting weights from training the DRL agent will be saved in Model_training_weights folder.
- `Model_training_logs`:    The resulting logs from training the DRL agent will be saved in Model_training_logs folder.

## Setting up Enviornment:


- Install Conda:       If conda is not installed in your system then install conda. (I used 4.10.1) <br>
- Install git:         - If git is not installed in system then install git ( I used 2.40.0.windows.1)<br>
                        - Download git from  https://git-scm.com/downloads and install it. <br>
                        - While installing: On the "Adjusting your PATH environment" page, select the option "Git from the command line and also from 3rd-party software." This ensures that Git is added to your system's PATH 
                          Environment variable, allowing you to use Git from the command prompt or terminal.<br>
                        - Verify the installation:  After the installation is complete, open a new command prompt or terminal window and run the following command to verify the Git version:

   ```shell
   git --version
- Clone the repository: Run the following command in your command prompt or terminal to clone the GitHub repository to your local system:

     ```shell
     git clone https://github.com/talhazaidi13/Cascaded-Deep-Reinforcement-Learning-Based-Multi-Revolution-Low-Thrust-Spacecraft-Orbit-Transfer.git
     ```
   Alternatively, you can just download the code files from the above link. 
            
- Navigate to the project directory:  Navigate to the project directory on your local system, which contains the cloned repository. In that folder you will find the environment.yml file. you can use the cd command 
                                       to navigate to the folder. e.g if environment.yml is at the location of D:\project then
   ```shell   
   cd "D:\project"
- Create conda environment: create the Conda environment using the environment.yml file. use the following code: <br>
   ```shell
   conda env create -f environment.yml  
   ```
   This command will create a new Conda environment based on the specifications in the environment.yml file. <br>
- Activate the environment: Use the following command to activate the environment: 
   ```shell                                        
   conda activate mat_py3.7  
   ```
   Please note that the name mat_py3.7 is the name of environment specified in the enviornment.yml file. You can rename it according to you.  <br>


This will create the desired environment in your local system. However we need to install MATLAB and Matlab python engine in our enviornment, through which python will communicate with MATLAB. Please follow the following steps for this. <br>
 
1. Install MATLAB: Install MATLAB on your system. (I am using MATLAB 2021a). If you dont have matlab, you can use the following link to  install MATLAB <br> https://www.mathworks.com/products/new_products/previous_release_overview.html <br>
2. Activate the Conda environment: Activate your Conda environment by running the following command in your command prompt or terminal:
   ```shell
   conda activate <environment_name>
   ```
3. Navigate to the MATLAB folder: In the activated Conda environment, go to the MATLAB folder by running the following command:
   ```shell   
   cd "<MATLAB_installation_folder>"  
   ```
   Replace <MATLAB_installation_folder> with the path to your MATLAB installation folder. By default, the MATLAB folder is located at "C:/Program Files/MATLAB". Make sure to include the double quotes if the path contains 
   spaces.
4. Go to the MATLAB Engine Python folder: Change the directory to the MATLAB Engine Python folder by running the following command:
   ```shell
   cd "R2021a\extern\engines\python"  
   ```
   This will navigate you to the relevant folder containing the MATLAB Engine Python setup file.
5. Install the MATLAB Engine: To install the MATLAB Engine in your Conda environment, execute the setup.py file by running the following command:
   ```shell
   python setup.py install  
   ```
   This command will install the MATLAB Engine package in your Conda environment.
6. Verify the installation: To check if the MATLAB Engine is installed correctly, run the following command:
   ```shell
   python -c "import matlab.engine" 
   ```

## Running the code:

Before running the code, please ensure that you have set the paths for the CSV files. The CSV files serve as the communication link between the MATLAB environment's step function and the Python code. Without correctly setting up the paths, the state values will not be updated.

- To set the paths for the CSV files, follow these steps:

1. Open the Mat_env.m file.
2. Locate lines #126 and #184 in the Mat_env.m file.
3. In those lines, modify the path for the csvlist.dat file to match the location of the file on your system. <br>
   For example, if the location of the csvlist.dat file on your system is D:/Cascaded-DRL/csv_files/csvlist.dat, update the lines as follows:
   ```python
   M = csvread('D:/Cascaded-DRL/csv_files/csvlist.dat')
   ```
  Replace D:/Cascaded-DRL/csv_files/csvlist.dat with the actual path to the csvlist.dat file on your system.

-  Open Git Bash from the Start menu and Activate your Conda environment by running the appropriate command. For example, if your Conda environment is named "mat_py3.7," you can use the following command:
 ```shell
conda activate mat_py3.7
```

- Change the current directory to the folder containing the test.sh or train.sh file using the cd command. For example, if the test.sh file is located D:/Cascaded-DRL, you can use the following command:
```shell
cd "D:/Cascaded-DRL"
```
- Finally, you can run the test.sh or train.sh file  for testing with trained weights and training from the scratch, using the bash command:
```shell
bash test.sh
bash train.sh
```

## Results
CDRL based GTO to GEO transfer  | CDRL based Super-GTO to GEO transfer
:-: | :-:
<image src='/paper-outputs/fig7.PNG' width=500/> | <image src='/paper-outputs/fig13.PNG' width=500/>
<image src='/paper-outputs/tab3.PNG' width=500/> | <image src='/paper-outputs/tab6.PNG' width=500/>
<image src='/paper-outputs/fig5.PNG' width=500/> | <image src='/paper-outputs/fig10.PNG' width=500/>
<image src='/paper-outputs/fig8.PNG' width=500/> | <image src='/paper-outputs/fig11.PNG' width=500/>
<image src='/paper-outputs/fig6.PNG' width=500/> | <image src='/paper-outputs/fig12.PNG' width=500/>


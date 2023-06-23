# Cascaded-Deep-Reinforcement-Learning-Based-Multi-Revolution-Low-Thrust-Spacecraft-Orbit-Transfer
We provide the code repository for our paper This repository includes the necessary code to replicate our experiments and utilize our DRL model for spacecraft trajectory planning. By accessing the repository, researchers and practitioners can benefit from our approach to efficiently transfer spacecraft to GEO using low-thrust propulsion systems.

1. Install Conda:       If conda is not installed in your system then install conda. I am using conda 4.10.1 in my system. <br>
2. Install git:         - If git is not installed in system then install git ( I used 2.40.0.windows.1)<br>
                        - Download git from  https://git-scm.com/downloads and install it. <br>
                        - While installing: On the "Adjusting your PATH environment" page, select the option "Git from the command line and also from 3rd-party software." This ensures that Git is added to your system's PATH 
                          Environment variable, allowing you to use Git from the command prompt or terminal.<br>
                        - Verify the installation:  After the installation is complete, open a new command prompt or terminal window and run the following command to verify the Git version:

   ```shell
   git --version
3. Clone the repository: Run the following command in your command prompt or terminal to clone the GitHub repository to your local system:

   ```shell
   git clone https://github.com/talhazaidi131313/Cascaded-Deep-Reinforcement-Learning-Based-Multi-Revolution-Low-Thrust-Spacecraft-Orbit-Transfer.git

  
  Alternatively, you can just download the code files from the above link. 
            
4. Navigate to the project directory:  Navigate to the project directory on your local system, which contains the cloned repository. In that folder you will find the environment.yml file. you can use the following command 
                                       to navigate to this folder. 
   ```shell   
   cd "local path conatins the clonned repository containing environment.yml file "
5. Create conda environment: create the Conda environment using the environment.yml file. use the following code: <br>
   ```shell
   conda env create -f environment.yml  
  
This command will create a new Conda environment based on the specifications in the environment.yml file.
6. Activate the environment: Use the following command to activate the environment: 
   ```shell                                        
   conda activate mat_py3.7  <br>
 Please note that the name mat_py3.7 is the name of environment specified in the enviornment.yml file. You can rename it according to you.  <br>

This will create the desired environment in your local system. However we need to install MATLAB and Matlab python engine in our enviornment, through which python will communicate with MATLAB. Please follow the following steps for this. <br>
 
1. Install MATLAB: Install MATLAB on your system. (I am using MATLAB 2021a). If you dont have matlab, you can use the following link to  install MATLAB. https://www.mathworks.com/products/new_products/previous_release_overview.html
2. Activate the Conda environment: Activate your Conda environment by running the following command in your command prompt or terminal:
   ```shell
   conda activate <environment_name>   <br>
3. Navigate to the MATLAB folder: In the activated Conda environment, go to the MATLAB folder by running the following command:
   ```shell   
   cd "<MATLAB_installation_folder>"  <br>
Replace <MATLAB_installation_folder> with the path to your MATLAB installation folder. By default, the MATLAB folder is located at "C:/Program Files/MATLAB". Make sure to include the double quotes if the path contains 
   spaces.
4. Go to the MATLAB Engine Python folder: Change the directory to the MATLAB Engine Python folder by running the following command:
   ```shell
   cd "R2021a\extern\engines\python"   <br>
This will navigate you to the relevant folder containing the MATLAB Engine Python setup file.
5. Install the MATLAB Engine: To install the MATLAB Engine in your Conda environment, execute the setup.py file by running the following command:
   ```shell
   python setup.py install     <br>
This command will install the MATLAB Engine package in your Conda environment.
6. Verify the installation: To check if the MATLAB Engine is installed correctly, run the following command:
   ```shell
   python -c "import matlab.engine"  <br>


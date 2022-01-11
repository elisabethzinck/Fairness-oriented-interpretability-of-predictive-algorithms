# Fairness Oriented Interpretability of Predictive Algorithms
*Write introduction here*

## biasbalancer 
*Write about what BiasBalancer can do here*. The documentation of biasbalancer is found [here](https://elisabethzinck.github.io/Fairness-oriented-interpretability-of-predictive-algorithms/html/biasbalancer.html). The tutorial notebook (located in [`src_biasbalancer/biasbalancer_tutorial.ipynb`](https://github.com/elisabethzinck/Fairness-oriented-interpretability-of-predictive-algorithms/blob/main/src_biasbalancer/biasbalancer_tutorial.ipynb)) gives a brief introduction to biasbalancer and showcases the use of the toolkit using the classical COMPAS dataset. 

### Installation
All source code for biasbalancer is found in the folder `src_biasbalancer`. Biasbalancer can be installed as a package in the following way: 

- Step 0: Install python3 (preferably using the Anaconda Distribution). 
- Step 1: Download the repository either by cloning the repository or downloading it as a zip file (and unzip the folder). 
    - Note: When unzipping the files on a Windows computer, an warning may indicate that some files have too long file names. These files can be skipped. 
- Step 2: Create a virtual python environment. This is done by opening a terminal in the folder `src_biasbalancer` and running the following command `conda env create --name <your_env_name> --file requirements.txt` in the terminal, where you replace `<your_env_name>` with the desired name of the environment. This will install the packages required by biasbalancer. 
- Step 3: Activate environment with command `conda activate <your_env_name>`. 
- Step 4: BiasBalancer is only installable via pip, so to install BiasBalancer, run the command `pip install -e .` (note the `.` in the end of the command).
- Step 5: Check that biasbalancer works by running `import biasbalancer` in python. 

### Documentation


## Thesis Report


### Reproducibility 
*To do: Write how to reproduce results in report here*. 


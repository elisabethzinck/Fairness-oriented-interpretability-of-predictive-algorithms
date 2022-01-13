# Fairness Oriented Interpretability of Predictive Algorithms
This repository contains the fairness analysis toolkit `biasbalancer` and all code needed to reproduce the results in the thesis *Fairness Oriented Interpretability of Predictive Algorithms* by Caroline Amalie Fuglsang-Damgaard and Elisabeth Zinck finalizing their degree *Mathematical Modelling and Computation* at Technical University of Denmark (DTU). 

## BiasBalancer 
BiasBalancer is a toolkit for fairness analysis of a binary classifier. It facilitates nuanced fairness analyses taking several fairness criteria into account, enabling the user to get a fuller overview of the potential interactions between fairness criteria. The tutorial notebook (located in [`src_biasbalancer/biasbalancer_tutorial.ipynb`](https://github.com/elisabethzinck/Fairness-oriented-interpretability-of-predictive-algorithms/blob/main/src_biasbalancer/biasbalancer_tutorial.ipynb)) gives a brief introduction to biasbalancer and showcases the use of the toolkit using the canonical COMPAS dataset. 

BiasBalancer consists of three levels, where each level increasingly nuances the fairness analysis. 

 - The first level calculates a unified assessment of unfairness, taking the severity of false positives relative to false negatives into account. For explanation of the computed quantity see [level_1 Documentation](https://elisabethzinck.github.io/Fairness-oriented-interpretability-of-predictive-algorithms/html/biasbalancer.html#biasbalancer.balancer.BiasBalancer.level_1).

 - The second level gives a comprehensive overview of disparities across sensitive groups, including a barometer quantifying violations of several fairness criteria. See [level_2 Documentation](https://elisabethzinck.github.io/Fairness-oriented-interpretability-of-predictive-algorithms/html/biasbalancer.html#biasbalancer.balancer.BiasBalancer.level_2).

 - The third level includes several methods enabling further investigation into potential unfairness identified in level two. See [level_3 Documentation](https://elisabethzinck.github.io/Fairness-oriented-interpretability-of-predictive-algorithms/html/biasbalancer.html#biasbalancer.balancer.BiasBalancer.level_2) for information about the specific analyses.


The documentation of biasbalancer is found [here](https://elisabethzinck.github.io/Fairness-oriented-interpretability-of-predictive-algorithms/html/biasbalancer.html). 

### Installation
All source code for biasbalancer is found in the folder [`src_biasbalancer`](https://github.com/elisabethzinck/Fairness-oriented-interpretability-of-predictive-algorithms/blob/main/src_biasbalancer). Biasbalancer is installed as a package in the following way: 

- Step 0: Install python3 (preferably using the Anaconda Distribution). 
- Step 1: Download the repository either by cloning the repository or downloading it as a zip file (and unzip the folder). 
    - Note: When unzipping the files on a Windows computer, a warning may indicate that some files have too long file names. These files can be skipped. 
- Step 2: Create a virtual python environment. This is done by opening a terminal in the folder `src_biasbalancer` and running the following command `conda env create --name <your_env_name> --file requirements.txt` in the terminal, where you replace `<your_env_name>` with the desired name of the environment. This will install the packages required by biasbalancer. 
- Step 3: Activate environment with command `conda activate <your_env_name>`. 
- Step 4: BiasBalancer is only installable via pip, so to install BiasBalancer, run the command `pip install -e .` (note the `.` in the end of the command).
- Step 5: Check that biasbalancer works by running `import biasbalancer` in python. 


## Thesis
All results in the thesis are produced using the code in this repository. 

### Datasets
We've analyzed predictive algorithms trained on five datasets: German Credit Score dataset (available [here](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)), Taiwanese credit scoring (available [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)), COMPAS dataset (available [here](https://github.com/propublica/compas-analysis)), Catalan redivism dataset (available [here](http://cejfe.gencat.cat/en/recerca/opendata/jjuvenil/reincidencia-justicia-menors/)), CheXpert dataset (available [here](https://stanfordmlgroup.github.io/competitions/chexpert/)), and CheXpert demographic data (available [here](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf)). All datasets, except the CheXpert dataset and CheXpert demographic data, are located in `/data/raw/` in the repository. 

The CheXpert dataset and CheXpert demographic dataset are publicly available, but must not be shared with others. The CheXpert dataset can be downloaded using the form at the bottom of the webpage https://stanfordmlgroup.github.io/competitions/chexpert/, and the CheXpert demographic data can be obtained [here](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf) after creating a login. 
Placing the datasets in the repository such that the folders are located in `/data/CheXpert/raw/CheXpert-v1.0-small/` and `/data/CheXpert/raw/chexpertdemodata-2` will make the existing scripts work. 

The scripts in the folder `src/data/` pre-processes the datasets and place the cleaned datasets in the folder `data/processed/` (and in the folder `data/CheXpert/processed` for the CheXpert demographic data). The CheXpert dataset consists of a large number of chest X-rays located in different folders and files. The DataModule "CheXpertDataModule" in `src/models/data_modules.py` can be used to access the image files. 

### Models
All code for creating and training the models for getting predictions for the fairness analyses is located in the folder `src/models/`. To get the predictions for the three example datasets, run the files `catalan_neural_network.py`, `german_neural_network.py`, `taiwanese_neural_network`, and `run_logistic_regressions.py`. The resulting predictions are placed in the folder `data/predictions/`. 

A number of different models were created for the CheXpert case study, which can be seen in the large number of bash scripts used to run the models. The best, and therefore chosen, model has been trained by running the bash script `bash_scripts/cheXpert_experiments/run_adam_dp=2e-1.sh`. To get the predictions from this model, first, run the bash script to train the model, followed by running the script `src/models/cheXpert_make_test_predictions.py`. The latter script will create the predictions on the test set using the model. 

### Fairness Analyses
The scripts used to perform the fairness analysis are the scripts `src/visualization_description/evaluater.py` and `src/evaluater_chexpert.py`. Moreover, all figures can be found in the `figures` folder. 



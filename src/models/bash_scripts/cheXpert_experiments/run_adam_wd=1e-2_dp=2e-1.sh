#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name --  (should be the same as in script)
#BSUB -J adam_wd=1e-2_dp=2e-1
### -- ask for number of cores (default: 1) -- 
#BSUB -n 8 
### --- ask for gpu ---
#BSUB -gpu "num=1"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- amount of memory per core/slot -- 
#BSUB -R "rusage[mem=10GB]"
### -- job gets killed if it exceeds xGB per core/slot -- 
#BSUB -M 10GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo logs/adam_wd=1e-2_dp=2e-1/%J.out 

echo "Starting bash script"

### Make logging directory
mkdir -p logs/adam_wd=1e-2_dp=2e-1/

### Activate environment
module load python3/3.8.11
source .venv/bin/activate

### Run python script
echo "----------------------------"
echo "--- Output from Python -----"
echo "----------------------------"

# Running neural network training parsing 
# model_name, weight_decay, drop_out, extented_image_augmentation
python3 src/models/cheXpert_neural_network_w_argparser.py adam_wd=1e-2_dp=2e-1 1e-2 2e-1 0

echo "----------------------------"
echo "---       DONE :)      -----"
echo "----------------------------"


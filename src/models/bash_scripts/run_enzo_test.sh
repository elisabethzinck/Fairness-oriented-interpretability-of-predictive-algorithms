#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name --  (should be the same as in script)
#BSUB -J enzo_test
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
#BSUB -W 02:00 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo logs/enzo_test.out 

echo "Starting bash script"

### Activate environment
module load python3/3.8.11
source .venv/bin/activate

### Run python script
echo "----------------------------"
echo "--- Output from Python -----"
echo "----------------------------"

python3 src/models/cheXpert_neural_network.py

echo "----------------------------"
echo "---       DONE :)      -----"
echo "----------------------------"
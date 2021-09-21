#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J Simple_test
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- amount of memory per core/slot -- 
#BSUB -R "rusage[mem=128MB]"
### -- job gets killed if it exceeds xGB per core/slot -- 
#BSUB -M 128MB
### -- set walltime limit: hh:mm -- 
#BSUB -W 00:05 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo logs/run_simple_test.out 

echo "Starting bash script"

### Activate environment
module load python3/3.8.11
source .venv/bin/activate

### Run python script
echo "----------------------------"
echo "--- Output from Python -----"
echo "----------------------------"

python3 src/models/simple_test.py


echo "----------------------------"
echo "---       DONE :)      -----"
echo "----------------------------"
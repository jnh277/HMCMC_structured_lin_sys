#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=10:00:00           
#PBS -l software=torch
#PBS -k oe

source /etc/profile.d/modules.sh
module load pystan
 
cd /home/jnh277/HMCMC_structured_lin_sys

echo "Running"

python single_param_uniform_noise.py


#!/bin/bash

#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=10:00:00
#PBS -l software=torch
#PBS -k oe

source /etc/profile.d/modules.sh
module load pystan

cd /home/jnh277/HMCMC_structured_lin_sys

echo "Starting trial $T"

python run_unif_trial.py --save_file unif_trial_${T}


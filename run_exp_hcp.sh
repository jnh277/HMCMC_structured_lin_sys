#!/bin/bash

#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=10:00:00
#PBS -l software=torch
#PBS -k oe

source /etc/profile.d/modules.sh
module load pystan

cd /home/jnh277/HMCMC_structured_lin_sys


python
import pickle
import pystan as ps
stan_model = ps.StanModel(file="unknown_unif.stan")
# save compiled file
# save it to the file 'model.pkl' for later use
with open('unif_model.pkl', 'wb') as f:
    pickle.dump(stan_model, f)

exit()


python single_param_uniform_noise.py


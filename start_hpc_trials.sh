#!/bin/bash

# make executable with the following
# chmod u+x start_hpc_trials.sh
# run using ./start_hpc_trials.sh


for T in {1..200}
do
     # calls each job script
     echo "creating job for trial ${T}"
    qsub -v T=${T} run_exp_hcp.sh
done


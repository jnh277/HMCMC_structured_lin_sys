#!/bin/bash

# make executable with the following
# chmod u+x start_hpc_trials.sh
# run using ./start_hpc_trials.sh



#for T in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
for T in {1..5}
do
     # calls each job script
     echo "trial ${T}"
    # qsub -v T=${T} run_exp_trials.sh
done



import pystan as ps
import pickle

# on hpc run
# source /etc/profile.d/modules.sh
# module load pystan

# then run this file to compile

## ------ compile the stan model on the HPC

# compile stan model
stan_model = ps.StanModel(file="truncated_normal_known_std.stan")
# save compiled file
with open('trunc_normal_model_std_hcp.pkl', 'wb') as f:
    pickle.dump(stan_model, f)


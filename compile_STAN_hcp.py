
import pystan as ps
import pickle


## ------ compile the stan model on the HPC

# compile stan model
stan_model = ps.StanModel(file="studentT_model.stan")
# save compiled file
with open('studentT_model_hcp.pkl', 'wb') as f:
    pickle.dump(stan_model, f)


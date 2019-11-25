
import pystan as ps
import pickle





## ------ now use STAN to get a bayesian estimate ------------

# compile stan model
stan_model = ps.StanModel(file="unknown_unif.stan")
# save compiled file
# save it to the file 'model.pkl' for later use
with open('unif_model_hcp.pkl', 'wb') as f:
    pickle.dump(stan_model, f)


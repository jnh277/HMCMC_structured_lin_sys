import numpy as np
import pandas as pd
import pystan as ps
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path




## ------ now use STAN to get a bayesian estimate ------------

# compile stan model
stan_model = ps.StanModel(file="unknown_unif.stan")
# save compiled file
# save it to the file 'model.pkl' for later use
with open('unif_model.pkl', 'wb') as f:
    pickle.dump(stan_model, f)


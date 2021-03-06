import numpy as np
import pandas as pd
import pystan as ps
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import scipy


theta = 1.0
N = 1000
gamma = 0.25
y = scipy.stats.cauchy.rvs(loc=theta, scale=gamma, size=N)


plt.subplot(1,1,1)
plt.hist(y, density=True,bins=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
plt.xlabel('y val')
plt.ylabel('histogram density')
plt.show()

## ------ now use STAN to get a bayesian estimate ------------
save_file = Path("./cauchy_model.pkl")
if save_file.is_file():
    stan_model = pickle.load(open('cauchy_model.pkl', 'rb'))
else:
    # compile stan model
    stan_model = ps.StanModel(file="cauchy_model.stan")
    # save compiled file
    # save it to the file 'model.pkl' for later use
    with open('cauchy_model.pkl', 'wb') as f:
        pickle.dump(stan_model, f)



# Transform the data into a data dict of the structure that stan wants
# data_dict = {"y": y, "N": len(y), "min_y": y.min(), "max_y": y.max(), "eps": eps}
data_dict = {"y": y, "N": len(y)}

# initiate monte carlo sampling
control = {"adapt_delta": 0.8}
stan_fit = stan_model.sampling(data=data_dict, thin=2, control=control, iter=4000, chains=4)

print(stan_fit)

def plot_trace(param,num_plots,pos, param_name='parameter'):
    """Plot the trace and posterior of a parameter."""

    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

    # Plotting
    plt.subplot(num_plots, 1, pos)
    plt.hist(param, 30, density=True);
    sns.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--', label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--', label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()

plot_trace(stan_fit["theta"],2,1,"theta")
plot_trace(stan_fit["sigma"],2,2,"gamma")
plt.show()

# least squares estimate
A = np.ones((len(y),1))
Ainv = np.linalg.pinv(A)
theta_sq = np.matmul(Ainv, y)

print("stan error",stan_fit["theta"].mean()-theta)
print("least squares error",theta_sq - theta)



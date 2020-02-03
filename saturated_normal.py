import numpy as np
import pandas as pd
import pystan as ps
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path



N = 200
x = np.linspace(0,6,num=N)
theta = 0.5
U = 1.0
L = 0.45

sigma = 0.1

y = theta + np.random.normal(0, sigma, N)

count = sum(y > U)
count += sum(y < L)
y[y > U] = U
y[y < L] = L

plt.subplot(2, 1, 1)
plt.plot(y)
plt.ylabel('y val')
plt.xlabel('y number')

plt.subplot(2,1,2)
plt.hist(y, density=True)
plt.xlabel('y val')
plt.ylabel('histogram density')
plt.show()

## ------ now use STAN to get a bayesian estimate ------------
save_file = Path("./sat_normal_model.pkl")
if save_file.is_file():
    stan_model = pickle.load(open('sat_normal_model.pkl', 'rb'))
else:
    # compile stan model
    stan_model = ps.StanModel(file="saturated_normal.stan")
    # save compiled file
    # save it to the file 'trunc_normal_model.pkl' for later use
    with open('sat_normal_model.pkl', 'wb') as f:
        pickle.dump(stan_model, f)


# data_dict = {"y": y, "N": len(y), "eps": 1e-8, "U": U, "L": L, "sig2":sigma*sigma}
data_dict = {"y": y, "N": len(y), "eps": 1e-8, "U": U, "L": L}


control = {"adapt_delta": 0.85, "max_treedepth": 10}
stan_fit = stan_model.sampling(data=data_dict, thin=2, control=control, iter=4000, chains=4)

print(stan_fit)

A = np.ones((len(y),1))
Ainv = np.linalg.pinv(A)
theta_sq = np.matmul(Ainv, y)

print(stan_fit["theta"].mean()-theta)
print(theta_sq-theta)

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
plot_trace(stan_fit["sigma"],2,2,"sigma")
# plot_trace(stan_fit["L"],3,3,"Upper")
plt.show()
import numpy as np
import pandas as pd
import pystan as ps
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from scipy.stats import norm
import math


#### ---------- Lessons ---------------- ####
# Requires either:
#   known standard deviation
#   or known limits

N = 1000
theta = 0.5
U = 0.65
L = -5.0

sigma = 0.1


# sample from a truncated Gaussian
def real_normal_lub_rng(mu, sigma, lb, ub, N):
    p_lb = norm.cdf(lb, loc=mu, scale=sigma)
    p_ub = norm.cdf(ub, loc=mu, scale=sigma)
    u = np.random.uniform(low=p_lb, high=p_ub, size=N)
    x = mu + sigma * norm.ppf(u)
    return x


y =real_normal_lub_rng(theta, sigma, L, U, N)

count = sum(y > U)
count += sum(y < L)


plt.subplot(2, 1, 1)
plt.plot(y)
plt.ylabel('y val')
plt.xlabel('y number')

plt.subplot(2,1,2)
plt.hist(y, density=True, bins=30)
plt.xlabel('y val')
plt.ylabel('histogram density')
plt.show()

# ------ now use STAN to get a bayesian estimate ------------
save_file = Path("./trunc_normal_model_lim.pkl")
if save_file.is_file():
    stan_model = pickle.load(open('trunc_normal_model_lim.pkl', 'rb'))
else:
    # compile stan model
    stan_model = ps.StanModel(file="truncated_normal_known_limits.stan")
    # save compiled file
    # save it to the file 'trunc_normal_model.pkl' for later use
    with open('trunc_normal_model_lim.pkl', 'wb') as f:
        pickle.dump(stan_model, f)


data_dict = {"y": y, "N": len(y), "U": U, "L": L}

control = {"adapt_delta": 0.8}
stan_fit = stan_model.sampling(data=data_dict, thin=2, control=control, iter=4000, chains=4)

print(stan_fit)

# stan_fit.plot()
# plt.show()
# least squares estimate
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
plot_trace(stan_fit["sigma"],2,2,"variance")
plt.show()
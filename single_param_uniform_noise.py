import numpy as np
import pandas as pd
import pystan as ps
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import scipy.optimize as opt
import arviz


theta = 1.0
N = 30
eps = 0.3
y = np.random.uniform(low=theta-eps,high=theta+eps,size=N)  # uniform noisy measurements

# plt.subplot(2, 1, 1)
# plt.plot(y)
# plt.ylabel('y val')
# plt.xlabel('y number')
#
# plt.subplot(2,1,2)
# plt.hist(y, density=True)
# plt.xlabel('y val')
# plt.ylabel('histogram density')
# plt.show()


## ------ now use STAN to get a bayesian estimate ------------
save_file = Path("./unif_model.pkl")
if save_file.is_file():
    stan_model = pickle.load(open('unif_model.pkl', 'rb'))
else:
    # compile stan model
    stan_model = ps.StanModel(file="unknown_unif.stan")
    # save compiled file
    # save it to the file 'model.pkl' for later use
    with open('unif_model.pkl', 'wb') as f:
        pickle.dump(stan_model, f)



# Transform the data into a data dict of the structure that stan wants
# data_dict = {"y": y, "N": len(y), "min_y": y.min(), "max_y": y.max(), "eps": eps}
data_dict = {"y": y, "N": len(y), "eps": eps}

# initiate monte carlo sampling
control = {"adapt_delta": 0.85}
stan_fit = stan_model.sampling(data=data_dict, thin=3, control=control, iter=6000, chains=8)
# control=list(adapt_delta=0.85)
# extract the samples
# stan_results = pd.DataFrame(stan_fit.extract())
# print(stan_results.describe())
print(stan_fit)

def plot_trace(param, param_name='parameter'):
    """Plot the trace and posterior of a parameter."""

    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

    # Plotting
    plt.subplot(2, 1, 1)
    plt.plot(param)
    plt.xlabel('samples')
    plt.ylabel(param_name)
    plt.axhline(mean, color='r', lw=2, linestyle='--')
    plt.axhline(median, color='c', lw=2, linestyle='--')
    plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
    plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
    plt.title('Trace and Posterior Distribution for {}'.format(param_name))

    plt.subplot(2, 1, 2)
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
    plt.show()

theta = stan_fit["theta"]
eps = stan_fit["eps"]

plot_trace(theta,"theta")
# plot_trace(eps, "eps")


# least squares estimate
A = np.ones((len(y),1))
Ainv = np.linalg.pinv(A)
theta_sq = np.matmul(Ainv, y)
# def func(theta, x):
#     return theta
#
# theta_lsq = opt.leastsq(func,0.7,args=(0*y,y))

# stan_fit.plot()
# arviz.plot_trace(stan_fit)
# plt.show()
import numpy as np
import pandas as pd
import pystan as ps
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path


N = 15
n_coefs = 10
sigma = 0.05
n_coefs_est = 15

theta = np.random.normal(0, 1, n_coefs)
x = np.linspace(0,3,N)
f = np.zeros(np.size(x))

# for i in range(n_coefs):
#     f += np.power(x,i) * theta[i]

for i in range(n_coefs):
    f += theta[i]*np.sin(x*(i+1))


y = f + np.random.normal(0, sigma, N)

# x_val = np.linspace(0,3,N)
# f_val = np.zeros(np.size(x_val))
#
# for i in range(n_coefs):
#     f_val += theta[i]*np.sin(x_val*(i+1))





## ------ now use STAN to get a bayesian estimate ------------
save_file = Path("./lin_reg_horseshoe.pkl")
if save_file.is_file():
    stan_model = pickle.load(open('lin_reg_horseshoe.pkl', 'rb'))
else:
    # compile stan model
    stan_model = ps.StanModel(file="lin_reg_horseshoe.stan")
    # save compiled file
    # save it to the file 'trunc_normal_model.pkl' for later use
    with open('lin_reg_horseshoe.pkl', 'wb') as f:
        pickle.dump(stan_model, f)


# build data matrix
data_matrix = np.zeros((N, n_coefs_est))
# for i in range(n_coefs):
#     data_matrix[:,i] =  np.power(x,i)

for i in range(n_coefs_est):
    data_matrix[:,i] =  np.sin(x*(i+1))


data_dict = {"y": y, "n_obs": len(y), "n_coefs":n_coefs_est, "data_matrix":data_matrix}

control = {"adapt_delta": 0.8}
stan_fit = stan_model.sampling(data=data_dict, thin=2, control=control, iter=4000, chains=4)

print(stan_fit)
y_hat = stan_fit["y_hat"].reshape((4000,N)).mean(0)
theta_hat_samples = stan_fit["coefs"].reshape((4000,n_coefs_est))
theta_hat = theta_hat_samples.mean(0)

plt.subplot(1,1,1)
plt.plot(x,f)
plt.plot(x,y,'ro')
plt.plot(x,y_hat)
plt.show()


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

plot_trace(theta_hat_samples[:,0],3,1,"theta")
plot_trace(theta_hat_samples[:,1],3,2,"theta")
plot_trace(theta_hat_samples[:,13],3,3,"theta")
plt.show()


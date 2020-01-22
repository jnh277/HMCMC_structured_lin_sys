import numpy as np
import pickle
import os
import argparse
import scipy.io as sio
from scipy.stats import norm

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--save_file', default='', help='save file name (default: wont save)')


args = parser.parse_args()

N_data = [10, 17, 28, 46, 77, 129, 215, 359, 560, 1000]
# N_data = [10, 17]
# N_data = [30, 100, 200, 500, 1000]

def real_normal_lub_rng(mu, sigma, lb, ub, N):
    p_lb = norm.cdf(lb, loc=mu, scale=sigma)
    p_ub = norm.cdf(ub, loc=mu, scale=sigma)
    u = np.random.uniform(low=p_lb, high=p_ub, size=N)
    x = mu + sigma * norm.ppf(u)
    return x

sigma = 0.25
theta = 0.5
U = 1.0
L = 0.0
stan_model = pickle.load(open('trunc_normal_lim_hpc.pkl', 'rb'))
control = {"adapt_delta": 0.8}

theta_hat = np.zeros((len(N_data), 1))
sigma_hat = np.zeros((len(N_data), 1))
theta_ML = np.zeros((len(N_data), 1))

print('Running')

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

for nn in range(len(N_data)):
    N = N_data[nn]
    print('Inference with N = ', N)
    x = np.sin(np.linspace(0, 6.0, N)) + 1.0
    y = real_normal_lub_rng(theta * x, sigma, L, U, N)
    data_dict = {"y": y, "N": len(y), "L":L, "U":U, "x":x}
    with suppress_stdout_stderr():
        stan_fit = stan_model.sampling(data=data_dict, thin=2, control=control, iter=4000, chains=4)
    theta_hat[nn] = stan_fit["theta"].mean()
    sigma_hat[nn] = stan_fit["sigma"].mean()

    # least squares comparison
    A = np.ones((len(y), 1))
    A[:, 0] = x
    Ainv = np.linalg.pinv(A)
    theta_ML[nn] = np.matmul(Ainv, y)

print('Saving results')

# ----------------- save configuration options and results -------------------------------
if args.save_file is not '':
    data = vars(args)       # puts the config options into a dict
    data['theta_hat'] = theta_hat
    data['sigma_hat'] = sigma_hat
    data['N_data'] = N_data
    data['theta_ML'] = theta_ML
    sio.savemat('./results/'+ args.save_file+'.mat', data)

print('Finished')

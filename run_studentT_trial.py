import numpy as np
import pickle
import os
import argparse
import scipy.io as sio
import scipy
from scipy import stats

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--save_file', default='', help='save file name (default: wont save)')


args = parser.parse_args()

N_data = [10, 17, 28, 46, 77, 129, 215, 359, 560, 1000]
# N_data = [10, 17]
# N_data = [30, 100, 200, 500, 1000]

theta = 1.0
gamma = 0.25
df = 3.0


stan_model = pickle.load(open('studentT_model_hcp.pkl', 'rb'))
control = {"adapt_delta": 0.85}

theta_hat = np.zeros((len(N_data), 1))
gamma_hat = np.zeros((len(N_data), 1))
nu_hat = np.zeros((len(N_data), 1))
theta_ML = np.zeros((len(N_data), 1))
gamma_ML = np.zeros((len(N_data), 1))
nu_ML = np.zeros((len(N_data), 1))

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
    y = scipy.stats.t.rvs(df,loc=theta, scale=gamma, size=N)
    data_dict = {"y": y, "N": len(y)}
    with suppress_stdout_stderr():
        stan_fit = stan_model.sampling(data=data_dict, thin=2, control=control, iter=4000, chains=4)
    theta_hat[nn] = stan_fit["theta"].mean()
    gamma_hat[nn] = stan_fit["sigma"].mean()
    nu_hat[nn] = stan_fit["nu"].mean()

    # a maximum likelihood comparison
    MLE = stats.t.fit(y)
    theta_ML[nn] = MLE[1]
    nu_ML[nn] = MLE[0]
    gamma_ML[nn] = MLE[2]



print('Saving results')

# ----------------- save configuration options and results -------------------------------
if args.save_file is not '':
    data = vars(args)       # puts the config options into a dict
    data['theta_hat'] = theta_hat
    data['gamma_hat'] = gamma_hat
    data['nu_hat'] = nu_hat
    data['N_data'] = N_data
    data['theta_ML'] = theta_ML
    data['gamma_ML'] = gamma_ML
    data['nu_ML'] = gamma_ML
    sio.savemat('./results/'+ args.save_file+'.mat', data)

print('Finished')

import numpy as np
import pandas as pd
import pystan as ps
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path



N = 100
x = np.linspace(0,6,num=N)
xt = np.sin(x)
theta = 0.5
U = 5.0
L = - 0.5

sigma = 0.1

y = theta * xt + np.random.normal(0, sigma, N)

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


data_dict = {"y": y, "N": len(y), "eps": 1e-8, "xt": xt, "U": U, "L": L}

control = {"adapt_delta": 0.8}
stan_fit = stan_model.sampling(data=data_dict, thin=2, control=control, iter=4000, chains=4)

print(stan_fit)

stan_fit.plot()
plt.show()
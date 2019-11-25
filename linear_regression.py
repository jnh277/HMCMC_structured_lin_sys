# following the tutorial found here https://aidanrussell.com/2019/01/14/pystan-tutorial-1/

import numpy as np
import pandas as pd
import pystan as ps
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# create a dataset
df = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5], "y": [0.2, 1.4, 2.5, 6.1, 8.9, 9.7]})
# plot the dataset
df.plot(x="x", y="y", kind="scatter", color="r", title="Dataset to analyse")
plt.show()


## ------  first use statsmodel to get a Maximum likelihood estimate ------------------
# how would we approach a typical linear regression?
# we can try statsmodels
ols_model = smf.ols(formula="y ~ x + 1", data=df)
ols_fit = ols_model.fit()
print(ols_fit.summary())

# how can we demonstrate confidence in our results?
pred_df_ols = ols_fit.get_prediction().summary_frame(alpha=0.05)
pred_df_ols["ci_lower_diff"] = pred_df_ols["mean"] - pred_df_ols["mean_ci_lower"]
pred_df_ols["ci_upper_diff"] = pred_df_ols["mean_ci_upper"] - pred_df_ols["mean"]
plt.errorbar(
df["x"],
pred_df_ols["mean"],
yerr=pred_df_ols[["ci_lower_diff", "ci_upper_diff"]].T.values,
fmt="-",
capsize=4,
)
plt.scatter(df["x"], df["y"], c="r", zorder=2)
plt.title("OLS Fit Including 95% Confidence Interval")
plt.ylim([-2.5, 12.5])
plt.ylabel("y")
plt.xlim([-0.5, 5.5])
plt.xlabel("x")
plt.show()

## ------ now use STAN to get a bayesian estimate ------------
# load and compile stan model
stan_model = ps.StanModel(file="linear.stan")

# Transform the data into a data dict of the structure that stan wants
data_dict = {"x": df["x"], "y": df["y"], "N": len(df)}
# initiate monte carlo sampling
stan_fit = stan_model.sampling(data=data_dict)

# extract the samples
stan_results = pd.DataFrame(stan_fit.extract())
print(stan_results.describe())

# here is one way to visualise the stan result, including uncertainty
# this does not include any filtering to eg 95%, it simply shows all inferences
for row in range(0, len(stan_results)):
    fit_line = np.poly1d([stan_results["beta"][row], stan_results["alpha"][row]])
    x = np.arange(6)
    y = fit_line(x)
    plt.plot(x, y, "b-", alpha=0.025, zorder=1)
plt.scatter(df["x"], df["y"], c="r", zorder=2)
plt.title("All Stan Fits Together")
plt.ylim([0, 12])
plt.ylabel("y")
plt.xlim([0, 5])
plt.xlabel("x")
plt.show()

# produce the prediction for each sample that was drawn
pred_df_stan = stan_results.copy()
summary_stan = pd.DataFrame(columns=["y_025", "y_50", "y_975"], index=range(0, 6))
for x in range(0, 6):
    pred_df_stan["x"] = x
    pred_df_stan[f"y_{x}"] = (pred_df_stan["alpha"] + pred_df_stan["beta"] * pred_df_stan["x"])
    summary_stan.loc[x, f"y_025"] = pred_df_stan[f"y_{x}"].quantile(q=0.025)
    summary_stan.loc[x, f"y_500"] = pred_df_stan[f"y_{x}"].quantile(q=0.5)
    summary_stan.loc[x, f"y_975"] = pred_df_stan[f"y_{x}"].quantile(q=0.975)
# produce a chart in the style of the previous OLS confidence interval chart
summary_stan["ci_lower_diff"] = summary_stan["y_025"] - summary_stan["y_500"]
summary_stan["ci_upper_diff"] = summary_stan["y_500"] - summary_stan["y_975"]
plt.errorbar(
    summary_stan.index,
    summary_stan["y_500"],
    yerr=summary_stan[["ci_lower_diff", "ci_upper_diff"]].T.values,
    fmt="-",
    capsize=4,
    )
plt.scatter(df["x"], df["y"], c="r", zorder=2)
plt.title("Stan Fit Including 95% Credible Interval")
plt.ylim([-2.5, 12.5])
plt.ylabel("y")
plt.xlim([-0.5, 5.5])
plt.xlabel("x")
plt.show()


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

alpha  = stan_fit['alpha']
beta = stan_fit['beta']
sigma = stan_fit['sigma']

plot_trace(alpha, 'alpha')
plot_trace(beta, 'beta')
plot_trace(sigma, 'sigma')

stan_fit.plot()
plt.show()
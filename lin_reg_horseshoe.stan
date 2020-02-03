// Linear regression with a horseshoe prior

data {
    int<lower=0> n_coefs;
    int<lower=0> n_obs;
    matrix[n_obs, n_coefs] data_matrix;
    vector[n_obs] y;
}
parameters {
    vector[n_coefs] coefs;
    real<lower=0.0000001> coefs_hyperpriors[n_coefs];
    real<lower=0.0000001> shrinkage_param;
    real<lower=0.0000001> sigma;
}
model {
//  shrinkage_param ~ cauchy(0.0,1.0);
  coefs_hyperpriors ~ cauchy(0.0, 1.0);
  for (i in 1:n_coefs)
//    coefs[i] ~ normal(0.0, coefs_hyperpriors[i]^2*shrinkage_param);
        coefs[i] ~ normal(0.0, coefs_hyperpriors[i]^2*shrinkage_param^2);
  sigma ~ cauchy(0.0, 1.0);

  for (n in 1:n_obs) {
        y[n] ~ normal(data_matrix[n, :] * coefs, sigma);
    }
}
generated quantities {
    vector[n_obs] y_hat;
    for (n in 1:n_obs) {
        y_hat[n] = data_matrix[n, :] * coefs;
    }
}


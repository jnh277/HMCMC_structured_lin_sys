data {
int<lower=1> N;
  real y[N];
}
parameters {
  real theta;
  real<lower=0.00001> sigma;
}
//transformed parameters {
//    real min_y;
//    real max_y;
//    min_y = min(y);
//    max_y = max(y);
//}
model {
    // priors
    theta ~ normal(0, 100);
    sigma ~ inv_gamma(3,1);
    // model
    y ~ cauchy(theta, sigma);
}
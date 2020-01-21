data {
int<lower=1> N;
  real y[N];
}
parameters {
  real theta;
  real<lower=0.000001> sigma;
  real<lower=0.000001,upper=1000> nu;
}
//transformed parameters {
//    real nu;
//    nu = 1;
//}
model {
    // priors
    theta ~ normal(0, 10);
//    sigma ~ inv_gamma(3,1);
    sigma ~ gamma(2,1);
    nu ~ gamma(2,1);

    // model
    y ~ student_t(nu, theta, sigma);
}
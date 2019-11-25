data {
int<lower=1> N;
  real y[N];
//  real<lower=0> eps;
}
parameters {
  real theta;
  real<lower=0> eps;
}
transformed parameters {
    real min_y;
    real max_y;
    min_y = min(y);
    max_y = max(y);

}
model {
    // priors
    //  theta ~ uniform(min_y, max_y);
    theta ~ uniform(min_y, max_y);
//    eps ~ cauchy((max_y-min_y)/2, 1);
//    eps ~ cauchy((max_y-min_y)/2, (max_y-min_y)/20);
//    eps ~ normal((max_y-min_y)/2,(max_y-min_y)/4);
    eps ~ inv_gamma(1,(max_y-min_y)*2);
    // model
    for (n in 1:N)
        y[n] ~ uniform(theta - eps, theta + eps);
}
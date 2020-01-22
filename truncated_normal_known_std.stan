data {
int<lower=1> N;
    real y[N];
    real<lower=0> sig2;
}
parameters {
  real<lower=min(y),upper=max(y)> theta;
  real L; // assuming we don't know truncation points
  real U;

}
transformed parameters {
    real min_y;
    real max_y;
    min_y = min(y);
    max_y = max(y);

}
model {
    // priors
//    theta ~ uniform(min_y, max_y);
    theta ~ normal((min_y+max_y)/2, 1) T[min_y,max_y];
    L ~ normal(min_y,0.1);
    U ~ normal(max_y,0.1);
//    sig2 ~ inv_gamma(2, 0.2);

    // model
    for (n in 1:N)
        y[n] ~ normal(theta, sqrt(sig2)) T[L,U];
}



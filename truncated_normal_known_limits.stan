data {
    int<lower=1> N;
    real U;
    real L;
    real<lower=L,upper=U> y[N];
//    real<lower=0.000001> sig2;
}
parameters {
    real<lower=min(y),upper=max(y)> theta;
    real<lower=0.000001> sigma;

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
//    L ~ normal(min_y,0.1);
//    U ~ normal(max_y,0.1);
//    sig2 ~ inv_gamma(2, 0.2);
    sigma ~ inv_gamma(2, 0.2);

    // model
    for (n in 1:N)
        y[n] ~ normal(theta, sigma) T[L,U];
}



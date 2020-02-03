data {
    int<lower=1> N;
    real y[N];
}
parameters {
    real theta;
    real<lower=0.000001> sigma;
    real<lower=0.000001> eps;

}
transformed parameters {
}
model {
    // priors
    theta ~ cauchy(0,10);
    sigma ~ inv_gamma(2, 0.2);
    eps ~ inv_gamma(1,(max(y)-min(y))*2);
//    sigma ~ cauchy(0,1)T[0,];
    // model
    for (n in 1:N)
        y[n] ~normal(theta, sigma)T[theta-eps,theta+eps];
}



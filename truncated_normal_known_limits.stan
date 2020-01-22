data {
    int<lower=1> N;
    real U;
    real L;
    real x[N];
    real<lower=L,upper=U> y[N];
}
parameters {
    real theta;
    real<lower=0.000001> sigma;

}
transformed parameters {
}
model {
    // priors
    theta ~ cauchy(0,10);
    sigma ~ inv_gamma(2, 0.2);
//    sigma ~ cauchy(0,1)T[0,];
    // model
    for (n in 1:N)
        y[n] ~normal(theta*x[n], sigma)T[L,U];
//     target += normal_lpdf(y | theta, sigma);
//     target += uniform_lpdf(y | L, U);

//    for (n in 1:N)
//        y[n] ~normal(theta, sigma)T[l,U];
//    for (n in 1:N){
//        y[n] ~ normal(theta, sigma);
//        y[n] ~ uniform(L,U);
//        }
}



functions {
    real norm_pdf(real y, real mu, real sig2){
        return exp(-0.5*(y - mu)*(y-mu)/sig2)/sqrt(2*pi()*sig2);
    }
    real norm_lpdf(real y, real mu, real sig2){
        return -log(sqrt(2*pi()*sig2)) -0.5*(y - mu)*(y-mu)/sig2;
    }
    real norm_cdf(real x, real mu, real sig2){
        return 0.5*(1+erf((x-mu)/sqrt(2*sig2)));
    }

}

data {
    int<lower=1> N;
    real y[N];
    real eps;
//    real<lower=eps> sig2;
    real L;
    real U;
}
parameters {
    real theta;
    real<lower=0.00001> sigma;
}
transformed parameters {
    real t1;
    real t2;
    real t3;
    t1 = normal_cdf(L, theta, sigma);
    t2 = 1 - normal_cdf(U, theta, sigma);
    t3 = 1-t1-t2;
}
model {
    // priors
    theta ~ cauchy(0.25, 1);
    sigma ~ gamma(2, 0.1);

    // model
    for (n in 1:N) {
        target += log_sum_exp(norm_lpdf(y[n] | theta, sigma*sigma) + log(t3),
                    norm_lpdf(y[n]|L,0.00000001)+log(t1));
    }
}
functions {
    real norm_pdf(real y, real mu, real sig2){
        return exp(-0.5*(y - mu)*(y-mu)/2/sig2)/sqrt(2*pi()*sig2);
    }
    real norm_cdf(real x, real mu, real sig2){
        return 0.5*(1+erf((x-mu)/sqrt(2*sig2)));
    }

}

data {
    int<lower=1> N;
    real xt[N];
    real U;
    real L;
    real<lower=L,upper=U> y[N];
//    real y[N];
    real eps;

}
parameters {
    real theta;
    real<lower=eps> sig2;
//    real L;
//    real U;

}
// transformed parameters {
//    real t1[N];
//    real t2[N];
//    for (n in 1:N) {
//        t1[n] = 0.5*(1+erf((L-xt[n])/sqrt(2*sig2)));
//        t2[n] = 1 - 0.5*(1+erf((U-xt[n])/sqrt(2*sig2)));
//     }
//}
model {
    real t1;
    real t2;
    real t3;

    // priors
    theta ~ cauchy(0.25, 0.5)T[0,1];
    sig2 ~ inv_gamma(2, 0.2);
//    sig2 = 0.01
//    L ~ cauchy(min(y), 0.1);
//    U ~ cauchy(max(y),0.1);

    // model
    for (n in 1:N) {
//        t1 = norm_cdf(L, xt[n], sig2);
//        target += log_mix(t1, uniform_lpdf())
        if (y[n] < L+eps) {
            t1 = norm_cdf(L, theta*xt[n], sig2);
//            target += log(t1 +  (1-t1)*norm_pdf(y[n],xt[n],sig2));
            target += log(t1);
        } else if (y[n] > U-eps) {
            t2 = 1 - norm_cdf(U, theta*xt[n], sig2);
//            target += log(t2 + (1-t2)*norm_pdf(y[n],xt[n],sig2));
            target += log(t2);
        } else {
//            y[n] ~ normal(xt[n]*theta, sqrt(sig2));
            t1 = norm_cdf(L, theta*xt[n], sig2);
            t2 = norm_cdf(U, theta*xt[n], sig2);
            t3 = t2 - t1;
            target += normal_lpdf(y[n]| xt[n]*theta, sqrt(sig2)) + log(t3);
        }
    }
}
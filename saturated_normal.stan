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
    real y[N];
    real eps;
    real<lower=eps> sig2;

}
parameters {
    real theta;
    real L;
    real U;

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
    L ~ cauchy(min(y), 5);
    U ~ cauchy(max(y),5);

    // model
    for (n in 1:N) {
        if (y[n] < L+eps) {
            t1 = norm_cdf(L, theta, sig2);
            target += log(t1);
        } else if (y[n] > U-eps) {
            t2 = 1 - norm_cdf(U, theta, sig2);
            target += log(t2);
        } else {
            t1 = norm_cdf(L, theta, sig2);
            t2 = norm_cdf(U, theta, sig2);
            t3 = t2 - t1;
            target += normal_lpdf(y[n]| theta, sqrt(sig2)) + log(t3);
        }
    }
}
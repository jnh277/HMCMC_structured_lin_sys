clear all
clc

N = 100;
mu = 1;
sigma = 0.25;
nu = 1;
x = random('tLocationScale',mu,sigma,nu,[N 1]);


[mu_1,sigma_1] = ML_T(x, 1000, nu)


function [mu,sigma] = ML_T(x, max_iter, nu)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

mu_i = mean(x);
C_i = cov(x);
d = 1;      % data dimensions is 1
n = length(x);

for i = 1:max_iter
    mu_old = mu_i;
    C_old = C_i;
    s_i = (x - mu_i).^2/C_i;
    w_i = (nu + d)./(nu+s_i);
    mu_i = sum(w_i.*x)/sum(w_i);
    C_i = sum(w_i.*(x-mu_i).^2)/n;
    if (abs(mu_i - mu_old) < 1e-10) && (abs(C_i-C_old) < 1e-10)
        break;
    end
end
sigma = sqrt(C_i);
mu = mu_i;

end

function [mu,sigma] = MLnu_T(x, max_iter, nu)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

mu_i = mean(x);
C_i = cov(x);
d = 1;      % data dimensions is 1
n = length(x);

for i = 1:max_iter
    mu_old = mu_i;
    C_old = C_i;
    s_i = (x - mu_i).^2/C_i;
    w_i = (nu + d)./(nu+s_i);
    mu_i = sum(w_i.*x)/sum(w_i);
    C_i = sum(w_i.*(x-mu_i).^2)/n;
    if (abs(mu_i - mu_old) < 1e-10) && (abs(C_i-C_old) < 1e-10)
        break;
    end
end
sigma = sqrt(C_i);
mu = mu_i;

end

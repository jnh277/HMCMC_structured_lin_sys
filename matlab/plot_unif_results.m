clear all
clc

files = dir('../results/unif_trial*.mat');

for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
end

theta_true = 1;
eps_true = 0.3;

theta_hat = [r.theta_hat];
theta_sq = [r.theta_sq];
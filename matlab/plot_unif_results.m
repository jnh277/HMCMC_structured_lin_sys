clear all
clc

files = dir('../results/unif_trial*.mat');

for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
end

theta_true = 1;
eps_true = 0.3;

theta_hat = [r.theta_hat];
eps_hat = [r.eps_hat];
theta_sq = [r.theta_sq];

%%
N_data = r(1).N_data;
ind = 4;

figure(1)
clf 
subplot 131
h1 = histogram(theta_sq(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([theta_true theta_true],ylims,'--','LineWidth',2);
h3 = plot([mean(theta_sq(ind,:)) mean(theta_sq(ind,:))],ylims,'--','LineWidth',2);
hold off
xlims = get(gca,'XLim');
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
title(['least squares estimates with N = ' num2str(N_data(ind))])

subplot 132
h1 = histogram(theta_hat(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([theta_true theta_true],ylims,'--','LineWidth',2);
h3 = plot([mean(theta_hat(ind,:)) mean(theta_hat(ind,:))],ylims,'--','LineWidth',2);
hold off
set(gca,'XLim',xlims);
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
title(['Conditional mean estimates with N = ' num2str(N_data(ind))])

subplot 133
h1 = histogram(eps_hat(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([eps_true eps_true],ylims,'--','LineWidth',2);
h3 = plot([mean(eps_hat(ind,:)) mean(eps_hat(ind,:))],ylims,'--','LineWidth',2);
hold off
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
title(['Conditional mean uniform limit estiamtes with N = ' num2str(N_data(ind))])

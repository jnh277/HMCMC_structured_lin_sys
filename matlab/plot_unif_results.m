clear all
clc

files = dir('../results/unif_data/unif_trial*.mat');

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
ind = 10;

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
% title(['Least squares with N = ' num2str(N_data(ind))])
title(['Least squares'])

set(gca,'FontSize',16)


subplot 132
h1 = histogram(theta_hat(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([theta_true theta_true],ylims,'--','LineWidth',2);
h3 = plot([mean(theta_hat(ind,:)) mean(theta_hat(ind,:))],ylims,'--','LineWidth',2);
hold off
set(gca,'XLim',xlims);
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
% title(['Conditional mean estimates with N = ' num2str(N_data(ind))])
title(['Conditional mean theta'])
set(gca,'FontSize',16)

subplot 133
h1 = histogram(eps_hat(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([eps_true eps_true],ylims,'--','LineWidth',2);
h3 = plot([mean(eps_hat(ind,:)) mean(eps_hat(ind,:))],ylims,'--','LineWidth',2);
hold off
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
title(['Conditional mean epsilon'])
set(gca,'FontSize',16)


%% 
cond_var = mean((theta_hat - theta_true).^2,2);
lsq_var = mean((theta_sq - theta_true).^2,2);

l1 = 2*eps_true^2./double(N_data).^2;
l2 = eps_true^2/3./double(N_data);

figure(2)
clf
loglog(N_data,cond_var,'LineWidth',2)
hold on
loglog(N_data,lsq_var,'LineWidth',2)
loglog(N_data,l1,'--','LineWidth',2)
loglog(N_data,l2,'--','LineWidth',2)
hold off
title('Estimator variances')
xlabel('Number of measurements (N)')
ylabel('Variance of estiamted theta')
l = legend('cond mean','least squares','$\frac{2\epsilon^2}{N^2}$','$\frac{\epsilon^2}{3N}$');
set(l,'Interpreter','latex','FontSize',16)
set(gca,'FontSize',20)

clear all
clc

files = dir('../results/normal_data/trial*.mat');

for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
end

theta_true = 1;
sigma_true = 0.25;

theta_hat = [r.theta_hat];
sigma_hat = [r.sigma_hat];
theta_ML = [r.theta_ML];
sigma_ML = sqrt([]);

%%
N_data = r(1).N_data;
ind = 10;

figure(1)
clf 
subplot 131
h1 = histogram(theta_ML(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([theta_true theta_true],ylims,'--','LineWidth',2);
h3 = plot([mean(theta_ML(ind,:)) mean(theta_ML(ind,:))],ylims,'--','LineWidth',2);
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
h1 = histogram(sigma_hat(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([sigma_true sigma_true],ylims,'--','LineWidth',2);
h3 = plot([mean(sigma_hat(ind,:)) mean(sigma_hat(ind,:))],ylims,'--','LineWidth',2);
hold off
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
title(['Conditional mean uniform limit estiamtes with N = ' num2str(N_data(ind))])

%% 
cond_var = mean((theta_hat - theta_true).^2,2);
lsq_var = mean((theta_ML - theta_true).^2,2);

% l1 = 2*eps_true^2./double(N_data).^2;
% l2 = eps_true^2/3./double(N_data);

figure(2)
clf
loglog(N_data,cond_var,'LineWidth',2)
hold on
loglog(N_data,lsq_var,'LineWidth',2,'LineStyle','--')
% loglog(N_data,l1,'--','LineWidth',2)
% loglog(N_data,l2,'--','LineWidth',2)
hold off
title('Estimator variances')
xlabel('Number of measurements (N)')
ylabel('Variance of estiamted theta')
l = legend('cond mean','least squares','$\frac{2\epsilon^2}{N^2}$','$\frac{\epsilon^2}{3N}$');
set(l,'Interpreter','latex','FontSize',16)

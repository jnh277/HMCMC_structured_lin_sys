clear all
clc

files = dir('../results/studentT_data/trial*.mat');

for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
end

theta_true = 1;
gamma_true = 0.25;
nu_true = 3.0;

theta_hat = [r.theta_hat];
gamma_hat = [r.gamma_hat];
nu_hat = [r.nu_hat];
theta_ML = [r.theta_ML];
gamma_ML = [r.gamma_ML];
nu_ML = 1./[r.nu_ML];


%%
N_data = r(1).N_data;
ind = 10;

figure(1)
clf 
subplot 231
h1 = histogram(theta_ML(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([theta_true theta_true],ylims,'--','LineWidth',2);
h3 = plot([mean(theta_ML(ind,:)) mean(theta_ML(ind,:))],ylims,'--','LineWidth',2);
hold off
xlims = get(gca,'XLim');
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
title(['Maximum likelihood $\theta$'],'Interpreter','latex')
set(gca,'FontSize',16)



subplot 234
h1 = histogram(theta_hat(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([theta_true theta_true],ylims,'--','LineWidth',2);
h3 = plot([mean(theta_hat(ind,:)) mean(theta_hat(ind,:))],ylims,'--','LineWidth',2);
hold off
set(gca,'XLim',xlims);
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
title(['Conditional mean $\theta$'],'Interpreter','latex')
set(gca,'FontSize',16)

subplot 232
h1 = histogram(gamma_ML(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([gamma_true gamma_true],ylims,'--','LineWidth',2);
h3 = plot([mean(gamma_ML(ind,:)) mean(gamma_ML(ind,:))],ylims,'--','LineWidth',2);
hold off
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
title(['Maximum likelihood $\sigma$'],'Interpreter','latex')
set(gca,'FontSize',16)

subplot 235
h1 = histogram(gamma_hat(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([gamma_true gamma_true],ylims,'--','LineWidth',2);
h3 = plot([mean(gamma_hat(ind,:)) mean(gamma_hat(ind,:))],ylims,'--','LineWidth',2);
hold off
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
title(['Conditional mean $\sigma$'],'Interpreter','latex')
set(gca,'FontSize',16)

subplot 233
h1 = histogram(nu_ML(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([nu_true nu_true],ylims,'--','LineWidth',2);
h3 = plot([mean(nu_ML(ind,:)) mean(nu_ML(ind,:))],ylims,'--','LineWidth',2);
hold off
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
title(['Maximum likelihood $\nu$'],'Interpreter','latex')
set(gca,'FontSize',16)

subplot 236
h1 = histogram(nu_hat(ind,:),'Normalization','probability');
ylims = get(gca,'YLim');
hold on
h2 = plot([nu_true nu_true],ylims,'--','LineWidth',2);
h3 = plot([mean(nu_hat(ind,:)) mean(nu_hat(ind,:))],ylims,'--','LineWidth',2);
hold off
legend([h1,h2,h3],'Estimator density','True Value','Estimator mean')
title(['Conditional mean $\nu$'],'Interpreter','latex')
set(gca,'FontSize',16)

%% variance of location estimate
cond_var = mean((theta_hat - theta_true).^2,2);
ML_var = mean((theta_ML - theta_true).^2,2);

% l1 = 2*eps_true^2./double(N_data).^2;
% l2 = eps_true^2/3./double(N_data);

figure(2)
clf
loglog(N_data,cond_var,'LineWidth',2)
hold on
loglog(N_data,ML_var,'LineWidth',2,'LineStyle','--')
% loglog(N_data,l1,'--','LineWidth',2)
% loglog(N_data,l2,'--','LineWidth',2)
hold off
title('Estimator variances','Interpreter','latex')
xlabel('Number of measurements (N)','Interpreter','latex')
ylabel('Variance of estiamted theta','Interpreter','latex')
l = legend('cond mean','maximum likelihood');
set(l,'Interpreter','latex','FontSize',16)
set(gca,'FontSize',20)


%% variance of scale estimate

cond_var = mean((gamma_hat - gamma_true).^2,2);
ML_var = mean((gamma_ML - gamma_true).^2,2);

% l1 = 2*eps_true^2./double(N_data).^2;
% l2 = eps_true^2/3./double(N_data);

figure(3)
clf
loglog(N_data,cond_var,'LineWidth',2)
hold on
loglog(N_data,ML_var,'LineWidth',2,'LineStyle','--')
% loglog(N_data,l1,'--','LineWidth',2)
% loglog(N_data,l2,'--','LineWidth',2)
hold off
title('Estimator variances','Interpreter','latex')
xlabel('Number of measurements (N)','Interpreter','latex')
ylabel('Variance of estiamted sigma','Interpreter','latex')
l = legend('cond mean','maximum likelihood');
set(l,'Interpreter','latex','FontSize',16)
set(gca,'FontSize',20)


%% variance of nu estimate

cond_var = mean((nu_hat - nu_true).^2,2);
ML_var = mean((nu_ML - nu_true).^2,2);


figure(4)
clf
loglog(N_data,cond_var,'LineWidth',2)
hold on
loglog(N_data,ML_var,'LineWidth',2,'LineStyle','--')
% loglog(N_data,l1,'--','LineWidth',2)
% loglog(N_data,l2,'--','LineWidth',2)
hold off
title('Estimator variances','Interpreter','latex')
xlabel('Number of measurements (N)','Interpreter','latex')
ylabel('Variance of estiamted nu','Interpreter','latex')
l = legend('cond mean','maximum likelihood');
set(l,'Interpreter','latex','FontSize',16)
set(gca,'FontSize',20)
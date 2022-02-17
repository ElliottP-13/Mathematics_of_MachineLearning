clear; clc;

load("simple.mat")

%% Part 2
% Train the model on the data in simple.mat using ten hat functions and μ = 105. Plot and turn in the learned model (the
% function fit to the data) on the interval [0,2π].
% **************************************************************************************

params = hat_basis(0, 2 * pi, 10);
[~, M] = size(params);
mu = 10^(5);
func = @func_hat;
w = lsefit(x, t, params, func, mu);

x_test = (0:0.01:2* pi)';  % Sample points to look at function
[N, ~] = size(x_test);
Sig_test =  eval_basis(params, func, x_test);

y = Sig_test * w;

hold on
scatter(x, t);
plot(x_test, y);
title("Mu = " + mu);
hold off


%% Part 3
% Do the same for other values of the hyperparameter such as μ = 10 and μ = 1.

mus = [10^5, 100, 10, 1, 0.1, 0.001];
figure()
tiledlayout(3,2)
for mu = mus
    params = hat_basis(0, 2 * pi, 10);
    [~, M] = size(params);
    func = @func_hat;
    w = lsefit(x, t, params, func, mu);


    x_test = (0:0.01:2* pi)';  % Sample points to look at function
    [N, ~] = size(x_test);
    Sig_test =  eval_basis(params, func, x_test);

    y = Sig_test * w;

    nexttile
    hold on
    scatter(x, t);
    plot(x_test, y);
    title("Mu = " + mu);
    hold off
end


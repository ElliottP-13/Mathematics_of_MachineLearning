clear; clc;

load("simple.mat")

%% Part 2
% Train the model on the data in simple.mat using ten hat functions and μ = 105. Plot and turn in the learned model (the
% function fit to the data) on the interval [0,2π].
% **************************************************************************************

params = hat_basis(0, 2 * pi, 10);
[~, M] = size(params);
func = @func_hat;

mu = 10^(5);
w = lsefit(x, t, params, func, mu);

x_test = (0:0.01:2* pi)';  % Sample points to look at function
[N, ~] = size(x_test);
Sig_test =  eval_basis(params, func, x_test);

y = Sig_test * w;

figure();
hold on
scatter(x, t);
plot(x_test, y);
title("Mu = " + mu);
hold off


%% Part 3
% Do the same for other values of the hyperparameter such as μ = 10 and μ = 1.
% **************************************************************************************
% I notice that as mu becomes small the function gets a lot more
% regularized. So at high values of mu the function is overfit, while at
% low values of mu it is underfit and becomes a line.

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


%% Part 5
% Train the model on the data in simple.mat using ten gauss functions and μ = 105. Plot and turn in the learned model (the
% function fit to the data) on the interval [0,2π].
% **************************************************************************************

params = gauss_basis(0, 2 * pi, 10);
[~, M] = size(params);
func = @func_gauss;

mu = 10^(5);
w = lsefit(x, t, params, func, mu);

x_test = (0:0.01:2* pi)';  % Sample points to look at function
[N, ~] = size(x_test);
Sig_test =  eval_basis(params, func, x_test);

y = Sig_test * w;

figure();
hold on
scatter(x, t);
plot(x_test, y);
title("Mu = " + mu);
hold off


%% Part 6
% Do the same for other values of the hyperparameter such as μ = 10 and μ = 1.
% **************************************************************************************
% I notice that as mu becomes small the function gets a lot more
% regularized. So at high values of mu the function is overfit, while at
% low values of mu it is underfit and becomes a line.

mus = [10^5, 100, 10, 1, 0.1, 0.001];
figure()
tiledlayout(3,2)
for mu = mus
    params = gauss_basis(0, 2 * pi, 10);
    [~, M] = size(params);
    func = @func_gauss;
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


%% Extra 2
% Train the model on the data in simple.mat using ten *fourier* functions and μ = 105. Plot and turn in the learned model (the
% function fit to the data) on the interval [0,2π].
% **************************************************************************************
% Note, I set M to be big here as I thought it was very fun. 
params = fourier_basis(0, 2 * pi, 100);
[~, M] = size(params);
func = @func_fourier;

mu = 10^(5);
w = lsefit(x, t, params, func, mu);

x_test = (0:0.001:2* pi)';  % Sample points to look at function
[N, ~] = size(x_test);
Sig_test =  eval_basis(params, func, x_test);

y = Sig_test * w;

figure();
hold on
scatter(x, t);
plot(x_test, y);
title("Mu = " + mu);
hold off


%% Extra 3
% Do the same for other values of the hyperparameter such as μ = 10 and μ = 1.
% **************************************************************************************
% I notice that as mu becomes small the function gets a lot more
% regularized. So at high values of mu the function is overfit, while at
% low values of mu it is underfit and becomes a line. The overfitting is a
% lot more obvious in Extra 2 than in the others. It is pretty cool. 

mus = [10^5, 100, 10, 1, 0.1, 0.001];
figure()
tiledlayout(3,2)
for mu = mus
    params = fourier_basis(0, 2 * pi, 10);
    [~, M] = size(params);
    func = @func_fourier;
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

%% Part 8
% Fit the model with integer values of μ = 1 to 100, and using a Gaussian basis of M = 10 elements on the data in
% simple.mat. For each model, calculate the squared error for the observations in test.mat (therefore testing the
% model). Generate and turn in a plot with μ on the x-axis and the total squared model error on the test data along the
% y-axis
gauss_params = gauss_basis(0, 2 * pi, 10);
gauss_func = @func_gauss;

hat_params = hat_basis(0, 2 * pi, 10);
hat_func = @func_hat;

four_params = fourier_basis(0, 2 * pi, 10);
four_func = @func_fourier;

load("test.mat")

e_g = zeros(1, 100);
e_h = zeros(1, 100);
e_f = zeros(1, 100);

t_g = zeros(1, 100);
t_h = zeros(1, 100);
t_f = zeros(1, 100);

for mu = 1:100
    w_g = lsefit(x, t, gauss_params, gauss_func, mu);
    w_h = lsefit(x, t, hat_params, hat_func, mu);
    w_f = lsefit(x, t, four_params, four_func, mu);
    
    sig_g = eval_basis(gauss_params, gauss_func, test_x);
    sig_h = eval_basis(hat_params, hat_func, test_x);
    sig_f = eval_basis(four_params, four_func, test_x);
    
    y_g = sig_g * w_g;
    y_h = sig_h * w_h;
    y_f = sig_f * w_f;
    
    e_g(mu) = sum((y_g - test_t).^2);
    e_h(mu) = sum((y_h - test_t).^2);
    e_f(mu) = sum((y_f - test_t).^2);
    
    sig_g2 = eval_basis(gauss_params, gauss_func, x);
    sig_h2 = eval_basis(hat_params, hat_func, x);
    sig_f2 = eval_basis(four_params, four_func, x);
    
    y_g = sig_g2 * w_g;
    y_h = sig_h2 * w_h;
    y_f = sig_f2 * w_f;
    
    t_g(mu) = sum((y_g - t).^2);
    t_h(mu) = sum((y_h - t).^2);
    t_f(mu) = sum((y_f - t).^2);
end

x_axis = [1:100];
figure()
hold on
plot(x_axis, e_g);
plot(x_axis, e_h);
plot(x_axis, e_f);

% plot(x_axis, t_g);
% plot(x_axis, t_h);
% plot(x_axis, t_f);
hold off

legend('Test Gaussian','Test Hat', 'Test Fourier')


%% Part 9
% What value of μ, when trained on the data in simple.mat, performs best on the data in test.mat? How do you know?
% Explain the shape of the plot you generated in the previous step.
% **************************************************************************************
% Below we print the optimal mu value. I know it is optimal (for the test
% set) becaus it is the mu that minimizes the squared error on the test
% set. The shape is interesting, to the left of the minimum the model is
% underfit and has not learned the data. And to the right the model is
% overfitting to the training data, making it perform worse on the test
% data. 
[M,optimal_mu] = min(e_g);
optimal_mu


%% Part 10
% Repeat the process, now fixing μ = 13 and varying the number of basis elements from M = 1 to 100. Generate and
% turn in a plot with the number of basis elements on the x-axis and the error for the test data on the y-axis
% **************************************************************************************
% This is also intereresting. We see a similar phenomenon as with the mu.
% Which is cool because it isn't the same as regularization which is
% underfitting and overfitting with a term, but it is the propper amount of
% basis functions. Essentially too few basis functions and it can't express
% the target function fully, but too few basis functions and the model gets
% too flexible. The optimal M is 9. It is also cool how the fourier and hat
% basis wiggle a lot over M. I don't have any good ideas as to why this
% happens.
load("test.mat")

e_g = zeros(1, 100);
e_h = zeros(1, 100);
e_f = zeros(1, 100);

t_g = zeros(1, 100);
t_h = zeros(1, 100);
t_f = zeros(1, 100);

mu = 13;

for M = 1:100
    gauss_params = gauss_basis(0, 2 * pi, M);
    gauss_func = @func_gauss;

    hat_params = hat_basis(0, 2 * pi, M);
    hat_func = @func_hat;

    four_params = fourier_basis(0, 2 * pi, M);
    four_func = @func_fourier;
    
    w_g = lsefit(x, t, gauss_params, gauss_func, mu);
    w_h = lsefit(x, t, hat_params, hat_func, mu);
    w_f = lsefit(x, t, four_params, four_func, mu);
    
    sig_g = eval_basis(gauss_params, gauss_func, test_x);
    sig_h = eval_basis(hat_params, hat_func, test_x);
    sig_f = eval_basis(four_params, four_func, test_x);
    
    y_g = sig_g * w_g;
    y_h = sig_h * w_h;
    y_f = sig_f * w_f;
    
    e_g(M) = sum((y_g - test_t).^2);
    e_h(M) = sum((y_h - test_t).^2);
    e_f(M) = sum((y_f - test_t).^2);
    
    sig_g2 = eval_basis(gauss_params, gauss_func, x);
    sig_h2 = eval_basis(hat_params, hat_func, x);
    sig_f2 = eval_basis(four_params, four_func, x);
    
    y_g = sig_g2 * w_g;
    y_h = sig_h2 * w_h;
    y_f = sig_f2 * w_f;
    
    t_g(M) = sum((y_g - t).^2);
    t_h(M) = sum((y_h - t).^2);
    t_f(M) = sum((y_f - t).^2);
end

x_axis = [1:100];
figure()
hold on
plot(x_axis, e_g);
plot(x_axis, e_h);
plot(x_axis, e_f);

% plot(x_axis, t_g);
% plot(x_axis, t_h);
% plot(x_axis, t_f);
hold off
legend('Test Gaussian','Test Hat', 'Test Fourier')

[M,optimal_M] = min(e_g);
optimal_M


%% Kicks and Giggles
figure();
tiledlayout(2,2)

func = @func_fourier;
mu = 10^(5);

params = fourier_basis(0, 2 * pi, 100);
[~, M] = size(params);
w = lsefit(x, t, params, func, mu);

x_test = (0:0.001:2* pi)';  % Sample points to look at function
[N, ~] = size(x_test);
Sig_test =  eval_basis(params, func, x_test);

y = Sig_test * w;

nexttile
hold on
scatter(x, t);
plot(x_test, y);
title("M = " + M);
hold off

params = fourier_basis(0, 2 * pi, 1000);
[~, M] = size(params);
w = lsefit(x, t, params, func, mu);

x_test = (0:0.001:2* pi)';  % Sample points to look at function
[N, ~] = size(x_test);
Sig_test =  eval_basis(params, func, x_test);

y = Sig_test * w;

nexttile
hold on
scatter(x, t);
plot(x_test, y);
title("M = " + M);
hold off

params = fourier_basis(0, 2 * pi, 3);
[~, M] = size(params);
w = lsefit(x, t, params, func, mu);

x_test = (0:0.001:2* pi)';  % Sample points to look at function
[N, ~] = size(x_test);
Sig_test =  eval_basis(params, func, x_test);

y = Sig_test * w;

nexttile
hold on
scatter(x, t);
plot(x_test, y);
title("M = " + M);
hold off

params = fourier_basis(0, 2 * pi, 5000);
[~, M] = size(params);
w = lsefit(x, t, params, func, mu);

x_test = (0:0.0001:2* pi)';  % Sample points to look at function
[N, ~] = size(x_test);
Sig_test =  eval_basis(params, func, x_test);

y = Sig_test * w;

nexttile
hold on
scatter(x, t);
plot(x_test, y);
title("M = " + M);
hold off



%% lsefit.m
%
% <include>lsefit.m</include>
%
%% hat_basis.m
%
% <include>hat_basis.m</include>
%
%% func_hat.m
%
% <include>func_hat.m</include>
%
%% gauss_basis.m
%
% <include>gauss_basis.m</include>
%
%% func_gauss.m
%
% <include>func_gauss.m</include>
%
%% fourier_basis.m
%
% <include>fourier_basis.m</include>
%
%% func_fourier.m
%
% <include>func_fourier.m</include>
%
%% eval_basis.m
%
% <include>eval_basis.m</include>
%
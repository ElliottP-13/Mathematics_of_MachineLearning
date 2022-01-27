%% Lab 1
% This is just a lab setting up MATLAB and loading in some data
% here we have some initial setup
clear; clc;
load("data.mat");

%% Experiment 1
% Using a for loop, compute the sum of l_2 norms of the columns of matrix X
tic
s = 0;
for i = 1:1000000
    s = s + norm(X(:,i));
end
s
toc
%% Expriment 2
% Don't use the for loop.
% This took me a while because I don't know how MATLAB works,
% if there is a cleaner solution, please tell me :)
tic
t = sum(sqrt(sum(X.^2)))
toc


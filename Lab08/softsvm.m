%SOFTSVM    Learns an approximately separating hyperplane for the provided data.
% [w, b, xi] = softsvm( X, l, gamma )
%
% Input: 
% X : D x N matrix of data points
% l : N x 1 vector with class labels (+/- 1)
% gamma : scalar slack variable penalty
%
% Output:
% w : D x 1 vector normal to the separating hyperplane
% b : scalar offset
% xi : N x 1 vector of slack variables
%
% classify data using sign( X'*w + b )

function [w, b, xi] = softsvm( X, l, gamma )

[D,N] = size(X);

% construct H, f, A, b, and lb


gamma = 0.005;
% Quadratic Objective
H = spdiags([zeros(N,1); ones(D,1); 0] , 0, N + D + 1, N + D + 1);
% Linear Objective
f =[gamma * ones(N, 1); zeros(D,1); 0];  % gamma N times, then D+1 zeros for right shape
% Linear Innequality Constraints Ax <= b
L = spdiags(l, 0, N, N );
A = -1 * [speye(N), L*X', l];
b = -1 * ones(N, 1);
% Lower bounds
lb = [zeros(N, 1); -Inf(D+1,1)];

% Solve
x = quadprog( H, f, A, b, [], [], lb ); 

% distribute components of x into w, b, and xi:
xi = x(1:N);
w = x(N+1:N+D);
b = x(end);
end
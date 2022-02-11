% PRINcipal COMPonent calculator
%   Calculates the principal components of a collection of points.
% Input:
%   X - D-by-N data matrix of N points in D dimensions.
% Output:
%   W - A D-by-M matrix containing the M principal components of the data.
%   Z - A M-by-N matrix containing the latent variables of the data.
%   mu - A D-by-1 vector containing the mean of the data.
%   lambda - A vector containing the eigenvalues associated with the above principal components.
function [W, Z, mu, lambda] = princomp(X,M)
[D, N] = size(X);

mu = mean(X, 2);
Y = X-mu;

% Compute PCA Vectors
if D > N  % high dimensional case
    [W2, V] = eigs(Y' * Y, M);
    W = Y * W2;
    W = W ./ vecnorm(W);
else  % normal case
    [W, V] = eigs(Y * Y', M);
end

% Now we need to project X onto span(W)
P = W' \ (W' * W);  % projection matrix formula
Z = P' * Y;

lambda = diag(V);  % get vector of eigenvalues
lambda = lambda(1:M);  % chop to first M values
end


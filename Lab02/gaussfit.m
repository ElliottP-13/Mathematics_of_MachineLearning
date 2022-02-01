%% GAUSSian FIT
%   Assuming data was sampled from a Gaussian distribution, returns the most
%  likely parameters for the underlying distribution.
% Input:
%   X - A D-by-N matrix with observation locations in each column (thus the
%   	observations are in D-dimensions and there are N of them).
% Output:
%   mu - D-by-1 vector indicating the center of the Gaussian distribution.
%   sigma - Scalar indicating the standard deviation of the Gaussian distribution.

function [mu, sigma] = gaussfit(X)
    [D, N] = size(X);
    mu = (1/N) * sum(X, 2);
    
    variance = (1/(D * N)) * sum(sum((X - mu).^2), 2);  % compute variance
    sigma = sqrt(variance);
end

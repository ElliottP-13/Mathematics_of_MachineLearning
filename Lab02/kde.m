% Kernel Density Estimation
%   Samples the kernel density estimate of a probability distribution using the
%  data in X with Gaussian kernel of standard deviation h. Samples are calculated
%  for each location in C.
% Input:
%   X - A D-by-N matrix with observation locations in each column (thus the
%   	observations are in D-dimensions and there are N of them).
%   h - A number indicating the standard deviation of the Gaussian kernel used.
%   C - Locations to evaluate the estimated distribution. Hence D-by-M, where if 
%      M = 1 this function calculates the KDE at one location.
% Output:
%   E - Evaluation of the estimated distribution at each of M locations given by 
%      the input C. Should be returned as a column vector.

function [E] = kde(X, h, C)

end
% Kernel Density Estimation MAP
%   Produces an N-by-N matrix representing event probabilities over the city
%	of Chicago. The lats and lons provided indicate event positions by latitude/longitude.
%  These positions and the standard deviation sigma are passed into kde.m in order to sample
%  the kernel density estimation on a uniform N-by-N grid over the city.
% Input:
%  lats - Row vector of latitudes for some number of events, such as crimes.
%  lons - Row vector of the same length as lats with the latitudes of the same events.
%  sigma - Standard deviation used to create the kernel density estimation, assuming the 
%  	events with the given (lat,lon) pairs were sampled from a probability distribution over
% 		the city of Chicago.
%  N - Dimensions of the event 'heat map' to generate. For example, N = 100 means this function will
%  	return a 100-by-100 dimension 'map' of where the provided events take place around Chicago.
% Output:
%  S - An N-by-N matrix with a view of the kernel density estimate round Chicago, using the provided 
%  	event locations and the function kde.m.

function [S] = kdemap(lats, lons, sigma, N)
    XY = [lats; lons];
    X = linspace(41.6, 42.05, N);
    Y = linspace(-87.95, -87.5, N);
    S = zeros(N, N);
    for i = 1:N
        for j = 1:N
            S(i,j) = kde(XY, sigma, [X(j); Y(i)]);
        end
    end
    S = flipud(S');
end
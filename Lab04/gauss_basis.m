%% generate a GAUSSian BASIS of functions.
%%  Produces parameters for a 1D basis of Gaussians on an interval.
%% Inputs:
%%  a - Beginning of the interval.
%%  b - End of the interval.
%%  num - Number of elements to generate.
%% Outputs:
%%  params - Matrix with, in each column, the parameters of a basis element.

function [params] = gauss_basis(a, b, num)
    params = zeros(2, num);
    for n = 1:num
        params(1,n) = a + (n - 1)*(b - a)/(num - 1);
        params(2,n) = (b-a)/num;
    end
end
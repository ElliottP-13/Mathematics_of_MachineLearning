%% generate a HAT BASIS of functions.
%%  Produces parameters for a 1D basis of hat functions on an interval.
%% Inputs:
%%  a - Beginning of the interval.
%%  b - End of the interval.
%%  num - Number of elements to generate.
%% Outputs:
%%  params - Matrix with, in each column, the parameters of a basis element.

function [params] = hat_basis(a, b, num)
    params = zeros(2, num);
    spacing = (b - a)/(num - 1);
    for n = 1:num
        c = a + (n - 1)*spacing;
        params(1,n) = c - spacing;
        params(2,n) = c + spacing;
    end
end
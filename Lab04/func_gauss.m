function [v] = func_gauss(x, params)
    v = (1/(params(2)*sqrt(2*pi))) .* exp(-(x - params(1)).^2 ./ (2*params(2)^2));
end
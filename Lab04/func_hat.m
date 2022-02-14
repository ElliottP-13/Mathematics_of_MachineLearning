function [v] = func_hat(x, params)
    c = 0.5*(params(1) + params(2));
    v = (x < c) .* (x - params(1))./(c - params(1));
    v = v + (x >= c) .* (1 - (x - c)./(params(2) - c));
    v(x < params(1)) = 0;
    v(x > params(2)) = 0;
end
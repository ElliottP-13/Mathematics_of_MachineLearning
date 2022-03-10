function [v] = func_fourier(x, params)
    k = params(1);
    a = params(2);
    b = params(3);
    v = sin(k * (x - a) + b);
end



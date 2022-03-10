function [params] = fourier_basis(a, b, num)
%FOURIER_BASIS Summary of this function goes here
%   Detailed explanation goes here
    params = zeros(3, num);  % sin(k(x-a) + b)
    k = (2 * pi) / (b - a);  % so it covers full range a,b
    offset = (b - a + pi/2) / (2 * pi);  % offset for cos functions
    scalar = 0;
    for n = 1:num
        params(1,n) = scalar * k;
        params(2,n) = a;
        if mod(n, 2) == 0
            params(3,n) = 0;
        else
            scalar = scalar + 1;
            params(3,n) = pi/2;
        end
    end
end


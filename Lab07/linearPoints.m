function [X,l] = linearPoints(m,b)
    n = 1000;
    R = 20;
    x0 = 0; % Center of the circle in the x direction.
    y0 = 0; % Center of the circle in the y direction.
    % Now create the set of points.
    t = 2*pi*rand(n,1);
    r = R*sqrt(rand(n,1));
    x = x0 + r.*cos(t);
    y = y0 + r.*sin(t);
    
    X = [x';y'];
    l = 2 * (m * x > y) - 1;
end


%% K-Means
%  Separate data points into K clusters with no other information.
% Inputs:
%  X - D-by-N matrix of N points in D dimensions.
%  K - Integer number of clusters to detect. 
% Outputs:
%  mu - D-by-K matrix with the learned cluster centroids.
%  labels - Length N vector with integer (1, 2, ..., K) class assignments.

function [mu, labels] = km(X, K)
    [D, N] = size(X);
    mu = X(:, randperm(N, K));  % init k random centroids from set
    labels = ones(1, N);
    
    run = true;
    while run
        % E step
        next_assignment = ones(1, N);
        for i = 1:N  % for all datapoints
            dist = vecnorm(mu - X(:,i));  % get dist to centers
            [~, idx] = min(dist);  % get index of closest center
            next_assignment(i) = idx; % assign point i to cluster
        end
        
        % M step
        for j = 1:K  % for all clusters
            assigned = next_assignment == j; % logical of all points assigned to cluster j
            s = sum(X(:, assigned), 2);
            mu(:, j) = s / size(X(:, assigned), 2);
        end
        
        run = all(next_assignment == labels);
        labels = next_assignment;
        
    end
end
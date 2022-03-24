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
    d = zeros(K,N);
    
    run = true;
    while run
        % E step
        for i = 1:K  % for all clusters
            d(i,:) = vecnorm(X - mu(:,i));  % get dist to centers
        end
        [~, next_assignment] = min(d, [], 1);
        
        % M step
        for j = 1:K  % for all clusters
            mu(:, j) = mean(X(:, next_assignment == j), 2);
        end
        
        run = all(next_assignment == labels);
        labels = next_assignment;
        
    end
end
%% adaBOOST model LEARNer
%  Uses the AdaBoost algorithm to train a classifier on data.
% Inputs
%  X - D x N : Observations
%  t - N x 1 : class labels
%  M - The number of weak learners to include in the ensemble.
% Outputs
%  params - A matrix containing the parameters for the M weak learners.
%  alpha - A vector of weights used to combine the results of the
%    M weak learners.

function [params, alpha] = boostlearn(X, t, M)
    [D, N] = size(X);
    weights = (1 / N) * ones(N, 1);
    
    alpha = zeros(M, 1);
    params = zeros(3, M);
    
    for k = 1:M
        params(:, k) = weaklearn(X, t, weights);
        
        preds = weakeval(X, params(:,k));  % get model predictions
        correct = preds == t;  % get logical vector of ones we got right
        wrong = ~correct;
        
        eps = sum(weights(wrong)) / sum (weights);
        a = log((1-eps)/eps);
        alpha(k) = a;
        
        weights = weights .* exp(a * wrong);
    end
    
end

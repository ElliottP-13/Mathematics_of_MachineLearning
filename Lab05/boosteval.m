%% adaBOOST model EVALuator
%  Uses a trained AdaBoost algorithm to classify data.
% Inputs
%  X - Matrix with observations (in columns) to classify. (D x N)
%  params - Output of boostlearn.m (weak learner parameters).
%  alpha - Output of boostlearn.m (weak learner mixing coefficients).
% Outputs
%  C - A matrix with predicted class labels (-1 or 1) for the input
%    observations in X.

function [C] = boosteval(X, params, alpha)
   % who cares about memory, lets just store NxM matrix of all our predictions! lol
   [D, N] = size(X);
   [p, M] = size(params);
   
   preds = zeros(N, M);
   for k = 1:M
      preds(:,k) = weakeval(X, params(:,k));; 
   end
   
   scaled = preds * alpha;  % scales predictions by alpha, and sums them (dot product)
   C = 2 * (scaled > 0) - 1;
end

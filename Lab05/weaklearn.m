%% WEAK LEARNer
%  Trains a simple classifier which achieves at least 50 percent accuracy.
% Inputs
%  X - Matrix with observations (in columns) to train. (D x N)
%  t - a vector (N x 1) with a label (-1 or +1) for each of the N data points
%  v - (Optional) Column vector with data weight for each data point. (N x 1) Defaults to uniform weights.
% Outputs
%  params - Parameters for the weak trained model (column vector).


function [params] = weaklearn(X, t, v)
    
    X0 = X(:,t==1);
    X1 = X(:,t==-1);
    
    if nargin == 2
        W0 = ones(size(X0,2),1);
        W1 = ones(size(X1,2),1);
    else
        W0 = v(t==+1);
        W1 = v(t==-1);
    end
    
    best_d = 1;
    best_x = 0;
    best_err = inf;
    is_01 = 1;
    
    X = [X0, X1];
    W = [-W0; W1];
    for d = 1:size(X0,1)
        [~,IX] = sort(X(d,:));
        
        err = cumsum(W(IX));
        [min_cum,min_k] = min(err);
        best_01 = sum(W0) + min_cum;
        best_01_x = X(d,IX(min_k));
        
        err = cumsum(-W(IX));
        [min_cum,min_k] = min(err);
        best_10 = sum(W1) + min_cum;
        best_10_x = X(d,IX(min_k));
       
        if best_01 < best_err
            best_d = d;
            best_x = best_01_x;
            best_err = best_01;
            is_01 = 1;
        end
        
        if best_10 < best_err
            best_d = d;
            best_x = best_10_x;
            best_err = best_10;
            is_01 = 0;
        end
    end
    
    beta = zeros(size(X0,1),1);
    beta(best_d) = 1;
    params = [beta;-best_x];
    if is_01
        params = -params;
    end
end
clear; clc;

%% Part 3
% Use make cloud.m to generate training data sampled from two point clouds
% **************************************************************************************

[X, t] = make_cloud();

%% Part 4
% First, let’s establish the performance of a single weak learner: use weaklearn.m to classify the data directly (with
% uniform weights). Plot the data (for example, use scatter) and distinguish between classes, and whether we classify
% correctly, or not (e.g. ’x’ vs ’o’ for the true class, and blue versus red for correct/wrong classification). How well did this
% classifier perform? Could a different single linear classifier perform better on this data? If so, what would the best case
% look like?
% **************************************************************************************
% Honestly the weak learner does pretty well. It gets 64.7% correct which
% is impressive. I don't think it is possible for a linear classifier to do
% significantly better on this. The distribution of the 'donut hole' does
% not appear to be perfectly circular, so perhaps a linear model could do
% better if its slope is parallel to the 'major axis' of the hole elipse.
% Or it could do a tiny bit better just based on the random density of
% points there might be a line that has higher density so it would
% nominally be better. But in reality, neither of these methods are
% significantly better.

params = weaklearn(X, t);
preds = weakeval(X, params);

figure('Name', 'weak learner' );
plot_preds('weak learner', X, t, preds);  % helper function to do the plotting

%% Part 5
% Use boostlearn.m to classify the same data as above with M =5 weak classifiers. Generate and turn in a figure
% showing the classification result.
% **************************************************************************************
% This looks pretty fun. It is a fun shape and not one that I initially expected.
% But in hindsight, it makes sense given the nature of our weak learners. 

[params, alpha] = boostlearn(X, t, 5);
preds = boosteval(X, params, alpha);

figure('Name', 'Ensemble M=5' );
plot_preds('Ensemble M = 5 learner', X, t, preds);  % helper function to do the plotting

%% Part 6
% Try larger values of M in the range 1 to 100 and study how the classifier changes. Based on visual inspection of the
% results: At what point do you believe the classifier is sufficiently reflecting the data? At what point do you believe the
% classifier might be over-fitting?
% **************************************************************************************
% The ensemble classifier reaches peak performance right around 15-20 (it
% changes each run). So within that range is a pretty accurate
% representation, and beyond that it starts to overfit. I would pick my M
% to be 17 because that seems to have a good balance of accuracy to
% overfitting based on multiple trials.

figure('Name', 'Comparison of different M values' );
tiledlayout(3, 3);

Ms = [3 5 10 15 17 20 50 75 100];
for M=Ms
    [params, alpha] = boostlearn(X, t, M);
    preds = boosteval(X, params, alpha);

    nexttile
    plot_preds(strcat('Ensemble M = ',num2str(M), ' learner'), X, t, preds);  % helper function to do the plotting

end

%% Part 7
% Use make cloud.m to generate a second set of observations (validation data). Test values for M, systematically. That
% is, for varying values of M, train AdaBoost on the training data, and evaluate its performance on the validation data.
% Generate and turn in a plot with M on the x-axis and the misclassification rate for validation data on the y-axis. Describe
% the shape of this graph.
% **************************************************************************************
% The shape is interesting and quite unexpected. I expected the error rate
% to go back up, as it overfits the data. But it just levels off at around
% M=20. This means that it is learning most of the shape of the underyling
% function with about M=20, and the extra weak learners do not really
% effect the performance of the model at all. For our weak learner and this
% example dataset, it implies that too many weak learners does not really
% worsen the performance of the ensemble at all, as the extra ones are
% pretty much ignored. This is not true in all cases, but for our very
% specific application in this lab, this seems to hold. 

[X_test, t_test] = make_cloud();

misclassification_rate = zeros(1, 100);

for M = 1:100
    [params, alpha] = boostlearn(X, t, M);
    preds = boosteval(X_test, params, alpha);
    
    correct = preds == t_test;
    wrong = ~correct;
    
    misclassification_rate(M) = sum(wrong) / numel(t_test);
end

figure('Name', 'Misclassification Rate vs M')
plot(1:100, misclassification_rate);
title('Misclassification Rate vs M');
ylabel('Misclassification rate');
xlabel('M');


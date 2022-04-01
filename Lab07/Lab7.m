clc; clear;
%% Part 1
% Implement softsvm. In order to test, I made linear points which 
% classifies points based on a line. So it is fully linearly seperable. We
% see that it works (yay!). The colors are the different classes, and the
% line is our approximate decision boundary.

[X, l] = linearPoints(1,0);  % make points with boundary y=x

[w, b, xi] = softsvm(X,l, 0.005);

% plot the results
figure();
hold on;
scatter(X(1,:), X(2,:), 5, l);  % plot the points
plot(X(1,:), -w(1)/w(2) * X(1,:));
colormap(gca,'prism')
hold off;

clear; % get rid of testing junk
%% Part 2
% Load the CBCL dataset and apply the soft SVM classifier with a penalty γ = 0.005. Generate and turn in a visualization
% of w, as found by the SVM function, using the command imagesc(reshape(w, dims)) (here dims comes from the
% original data file). What does this picture of w represent? How do you interpret it?
% **************************************************************************************
% It looks like a face :0
% Here, this picture of w represents the weights of each pixel of the face.
% It could be interpreted as how important each part of the face is in the
% classification.

load('cbcl1.mat')
[w, b, xi] = softsvm(X,L, 0.005);

figure();
imagesc(reshape(w, dims))

%% Part 3
% Generate a plot of X'*w + b (overlay with and compare to the plot of L, the correct labels). What do the extremes
% (minimum/maximum) of this plot represent? Were any data points classified incorrectly, and how can you tell?
% **************************************************************************************

preds = X' * w + b;
figure();
hold on;
scatter(preds, L, 5);  % plot the points
colormap(gca,'prism')
hold off;

%% Part 4
% How can we determine that a data point was a support vector?
% **************************************************************************************
% If its value is on the 'wrong' side of zero or if it is very close to
% zero (being the closest to zero of any of the points without being negative).


%% Part 5
% Turn in two images corresponding to the extreme points of this plot (most positive/negative scoring column of X shown
% as images), and two more images corresponding to example support vectors from each class. Discuss what you
% observe!
% **************************************************************************************
% The ones labeled as -1 look very much like faces. This is quite clear.
% It is interesting that the support looks most like a face. I would have
% expected the minimum to look most like a face, and the support to be
% something that is harder to distinguish. The positive examples just look
% like noise to me. I have honestly no idea what it is. 

[~, min_idx] = min(preds);
[~, max_idx] = max(preds);

idx = 1:length(preds);
pos_supports = idx(and((preds < 0), (L > 0)));
pos_sup_idx = randsample(pos_supports, 1);

neg_supports = idx(and((preds > 0), (L < 0)));
neg_sup_idx = randsample(neg_supports, 1);

figure()
tiledlayout(2, 2);
% Minimum
y = X(:, min_idx);
nexttile;
imagesc(reshape(y, dims))
title('Minimum')
% max
y = X(:, max_idx);
nexttile;
imagesc(reshape(y, dims))
title('Maximum')
% negative support
y = X(:, neg_sup_idx);
nexttile;
imagesc(reshape(y, dims))
title('Negative Support')
% positive support
y = X(:, pos_sup_idx);
nexttile;
imagesc(reshape(y, dims))
title('Positive Support')

%% Part 6
% Load the 20 Newsgroups data set and apply the soft SVM with γ = 0.005
% **************************************************************************************
clear; % get rid of CBCL stuff
load('news.mat');
[w, b, xi] = softsvm(X,L, 0.005);


%% Part 7
% By examination of the vector w, which words are the most important for separating the two classes of documents?
% Which words are most distinctly space-related? What about cryptography-related? Give at least five important words
% for each case.
% **************************************************************************************
% The most important words are the ones with the most extreme values of w.
% So we give the top (10) words with the most extreme values of w for each
% category and display it below.

clc;
idx = (1:length(w))';
A = [w, idx];
B = sortrows(A, 1);

n = 10;
top_negative = B(1:n, 2);
top_positive = B(end-n+1:end, 2);

disp("Top Positive (space):");
disp(dict(top_positive, :));

fprintf("\n\n");

disp("Top Negative (cryptography):");
disp(dict(top_negative, :));

%% softsvm.m
%
% <include>softsvm.m</include>
%
%% linearPoints.m
%
% <include>linearPoints.m</include>
%
%% imgrid1.m
%
% <include>imgrid1.m</include>
%

clc; clear;
%% Part 1.1
% First, play around:
% (a) Load and run the MLP regression code. You have to use MATLAB with the Deep Learning Toolbox and running
% version 2019b or newer—MATLAB in the cloud will work, but interaction with the toolbox window might be slow.
% (b) Study the plot. What does “loss” seem to represent? Is it going down monotonically? Is there a difference between
% training and validation performance? Explain what you observe!
% (c) Change some of the optimization parameters: What happens if you pick an initial learning rate that is 10x, 100x
% bigger/smaller? Can you find out what an “Epoch” is? What does sgdm stand for? Replace sgdm with a different
% choice (e.g., adam with much bigger initial learning rate, e.g., 0.1), and see what happens.
% (d) Remove a layer of nodes / Add another layer of nodes / Change the number of nodes in each layer (e.g., 5, 20, or
% 50). What happens?
% (e) Repeat the above for the MLP classification code
% **************************************************************************************
% (a) OK. 
% (b) Loss seems to be just the RMSE (Root Mean Squared Error) or something
% related, ie. probably just MSE. It doesn't quite go down entirely
% monontonically but has some bumps sometimes. This is probably due to the
% momentum, or a bad batch.
% (c) Starting with lr = e-2 causes the training graph to have just one
% huge spike from overcorrecting at loss on order of e32 and then nothing,
% the table shows NaN, so possibly infinite error? Starting with small e-8
% learning rate causes it to learn slowly. Epoch is # times through data,
% sgdm is stochastic gradient descent. Adam converges very quickly. And
% does a good job.
% (d) I added two new layers and validation rmse was about 0.03 which is
% actually worse than before, so it probably gets stuck in a local optima.
% The same thing happens when I set the number of nodes in the first layer
% to 50, it gets stuck at rmse of 0.17.

%% Part 1.2
% First, play around:
% (a) Load and run the MLP regression code. You have to use MATLAB with the Deep Learning Toolbox and running
% version 2019b or newer—MATLAB in the cloud will work, but interaction with the toolbox window might be slow.
% (b) Study the plot. What does “loss” seem to represent? Is it going down monotonically? Is there a difference between
% training and validation performance? Explain what you observe!
% (c) Change some of the optimization parameters: What happens if you pick an initial learning rate that is 10x, 100x
% bigger/smaller? Can you find out what an “Epoch” is? What does sgdm stand for? Replace sgdm with a different
% choice (e.g., adam with much bigger initial learning rate, e.g., 0.1), and see what happens.
% (d) Remove a layer of nodes / Add another layer of nodes / Change the number of nodes in each layer (e.g., 5, 20, or
% 50). What happens?
% (e) Repeat the above for the MLP classification code
% **************************************************************************************
% (a) OK. 
% (b) The loss plot/accuracy is a lot more jumpy for classification. I am
% not sure why. This could be because they are computed discretely (you
% can't half correctly get the answer). But that was surprising. It kind of
% shark fins around a bit.
% (c) At e-4 learning rate, it essentially does not learn. Hovering at like
% 50% for a while. At e-9 practically nothing happens. It is just a flat
% curve. It does *slowly* go up in accuracy, and it is monotone (as far as
% I can see). But it progresses a lot slower. At 1000 epochs it is only at
% about 65%
% (d) I removed the 20 node layer. It actually seems to do pretty good,
% getting around 97% on validation.

%% Part 2
% Build/train a two-input–two-output regression network for Cartesian to polar conversion: n(x, y) ≈ cart2pol(x, y). Plot
% the learned radius and angle landscape to “see” how good the training works.
% **************************************************************************************
% The network does a pretty poor job of learning the radius. It struggles around 
% some lines for some reason (different each time). It kind of looks like a blob.
% The angle is not learned pretty well. At pi
% and -pi it has a hard time, because there is a discontinuity and around 0. 
% But overall, it looks like a fairly smooth gradient across the space.
% This is surprising as it is a fairly non-trivial relation to learn (arcsin / arccos). 

% setting up two hidden layers with 10 nodes, each, all fully connected
layers = [ sequenceInputLayer( 2 )
    fullyConnectedLayer(20)
    tanhLayer
    fullyConnectedLayer(10)
    tanhLayer
    fullyConnectedLayer(10)
    tanhLayer
    fullyConnectedLayer(2)
    regressionLayer
    ]

% n points in WxW square centered around 0
n = 1000;
W = 2;
x = rand(1,n) * W - W/2;
y = rand(1,n) * W - W/2;
XTrain = [x ; y];
[th, ro] = cart2pol(x, y);
YTrain = [th ; ro];

x = rand(1,n/10) * W - W/2;
y = rand(1,n/10) * W - W/2;
XVal = [x ; y];
[th, ro] = cart2pol(x, y);
YVal = [th ; ro];

% training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',2000,...
    'InitialLearnRate',1e-5, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ValidationData', {XVal,YVal} );

net = trainNetwork( XTrain, YTrain, layers, options );

plotX = -1:0.01:1;
plotY = -1:0.01:1;
[X, Y] = meshgrid(plotX, plotY);
XTest = [X(:)' ; Y(:)'];
[thv, rov] = cart2pol(X(:)' , Y(:)');

th = reshape(thv, size(X));
ro = reshape(rov, size(X));

preds = net.predict(XTest);
lthv = preds(1,:);
lrov = preds(2,:);

lth = reshape(lthv, size(X));
lro = reshape(lrov, size(X));

figure();
tiledlayout(2,2);

nexttile;
h = pcolor(plotX, plotY, ro);
set(h, 'EdgeColor', 'none');
title('True radius');

nexttile;
h = pcolor(plotX, plotY, lro);
set(h, 'EdgeColor', 'none');
title('Learned radius');

nexttile;
h = pcolor(plotX, plotY, th);
set(h, 'EdgeColor', 'none');
title('True theta');

nexttile;
h = pcolor(plotX, plotY, lth);
set(h, 'EdgeColor', 'none');
title('Learned theta');


%% Part 3
% Pick CBCL1 or news data, and build a binary classifier. Put randomly selected portions of the data (10%, each) away
% for validation and testing (network trains on 80% data, validates on 10% while training, and when done, you run the
% network on the 10% left out for testing). What percent correct can you achieve on the training/validation/testing data?
% How does this compare against SVM (train/test SVM on the same data partitions used for the network).
% Page 4
% **************************************************************************************
% Initially, the neural net performed pretty bad. I think it was too small.
% So I increased the number of nodes in the first layer and also increased
% the depth of the network. Now it performs pretty well. Not that much
% better than SVM so it is maybe not worth the complexity. 

clear;
load('cbcl1.mat')

ii = randperm(size(X,2));
trainIdx = ii(1: floor(0.80 * length(ii)));
valIdx = ii(floor(0.80 * length(ii)) + 1 : floor(0.90 * length(ii)));
testIdx = ii(floor(0.90 * length(ii))+1 : end);

XTrain = X(:, trainIdx);
YTrain = L(trainIdx)';
XVal = X(:, valIdx);
YVal = L(valIdx)';
XTest = X(:, testIdx);
YTest = L(testIdx)';

[w, b, xi] = softsvm(XTrain,YTrain', 0.005);
SVMpreds = (XTest' * w + b) > 0;

layers = [ sequenceInputLayer( size(X,1) )
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(70)
    tanhLayer
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(2)		% there are two classes, so two of these nodes
    softmaxLayer				% 
    classificationLayer			% these two are needed for classification output
 ]


% training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',2000,...
    'InitialLearnRate',1e-7, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ValidationData', {XVal, categorical(YVal)} );

net = trainNetwork( XTrain, categorical(YTrain), layers, options );
NETpreds = net.classify(XTest);

netAccuracy = sum(NETpreds == categorical(YTest)) / length(YTest)
svmAccuracy = sum(SVMpreds' == (YTest > 0)) / length(YTest)

%% softsvm.m
%
% <include>softsvm.m</include>
%
%% MLP Classification.m
%
% <include>MLP_classification.m</include>
%
%% MLP Regression.m
%
% <include>MLP_regression.m</include>
%
%% swissroll.m
%
% <include>swissroll.m</include>
%
%% plotroll.m
%
% <include>plotroll.m</include>
%
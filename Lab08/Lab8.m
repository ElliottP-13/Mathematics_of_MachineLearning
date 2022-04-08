clc; clear;
%% Part 1
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
% (a)


%% Part 2
% Build/train a two-input–two-output regression network for Cartesian to polar conversion: n(x, y) ≈ cart2pol(x, y). Plot
% the learned radius and angle landscape to “see” how good the training works.
% **************************************************************************************


% setting up two hidden layers with 10 nodes, each, all fully connected
layers = [ sequenceInputLayer( 2 )
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
YTrain = [th, ro];

x = rand(1,n/10) * W - W/2;
y = rand(1,n/10) * W - W/2;
XVal = [x ; y];
[th, ro] = cart2pol(x, y);
YVal = [th, ro];

% training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',2000,...
    'InitialLearnRate',1e-5, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ValidationData', {XVal,YVal} );

net = trainNetwork( XTrain, YTrain, layers, options );

[X, Y] = meshgrid(-1:0.01:1, -1:0.01:1);
testIn = [X(:) ; Y(:)];

preds = net(testIn);


%% Part 3
% Pick CBCL1 or news data, and build a binary classifier. Put randomly selected portions of the data (10%, each) away
% for validation and testing (network trains on 80% data, validates on 10% while training, and when done, you run the
% network on the 10% left out for testing). What percent correct can you achieve on the training/validation/testing data?
% How does this compare against SVM (train/test SVM on the same data partitions used for the network).
% Page 4
% **************************************************************************************
% 

load('cbcl1.mat')
ii = randperm(size(X,2));
trainIdx = ii(1: floor(0.80 * length(ii)));
valIdx = ii(floor(0.80 * length(ii)) + 1 : floor(0.90 * length(ii)));
testIdx = ii(floor(0.90 * length(ii))+1 : end);

XTrain = X(:, trainIdx);
YTrain = L(trainIdx);
XVal = X(:, valIdx);
YVal = L(valIdx);
XTest = X(:, testIdx);
YTest = L(testIdx);

[w, b, xi] = softsvm(XTrain,YTrain, 0.005);
SVMpreds = XTest' * w + b;

layers = [ sequenceInputLayer( size(X,1) )
    fullyConnectedLayer(10)
    tanhLayer
    fullyConnectedLayer(10)
    tanhLayer
    fullyConnectedLayer(2)		% there are two classes, so two of these nodes
    softmaxLayer				% 
    classificationLayer			% these two are needed for classification output
    ]


% training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',2000,...
    'InitialLearnRate',1e-5, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ValidationData', {XVal,YVal} );

net = trainNetwork( XTrain, YTrain, layers, options );
NETpreds = net.classify(XTest);

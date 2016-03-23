%% SOFTMAX TRAINING
%In this script we train a softmax regression model on the MNIST dataset

clear;
load train.mat
load test.mat

% we take only 10 000 elements at random to speed up computations
[trainX, idx] = datasample(train.images, 10000, 'replace', false);
trainy = train.labels(idx,:);
[testX, idx] = datasample(test.images, 10000, 'replace', false);
testy = test.labels(idx,:);

% normalise features
trainX_normalised = gradient_descent_preprocessing(trainX);

% build the objective function together with parameters for gradient descent
objFct = softmaxcost_builder(trainX_normalised, trainy);
startPt = zeros(size(trainX, 2), 10);
options.momentum = 0.5;
options.timeStep = 1e-6;
options.debugMode = 1;
options.numIters = 1500;

weights = gradient_descent(startPt, objFct, options);

predictions = softmax(testX*weights);
[prob, class_pred] = max(predictions, [], 2);

error = (sum((class_pred-1) ~=testy))*1/length(testy)*100;
%% SOFTMAX TRAINING
%In this script we train a softmax regression model on the MNIST dataset

clear;
load train.mat
load test.mat

error = zeros(1,5);
for i=1:5
    % we take only 10 000 elements at random to speed up computations
    [trainX, idx] = datasample(train.images, 10000, 'replace', false);
    trainy = train.labels(idx,:);
    [testX, idx] = datasample(test.images, 10000, 'replace', false);
    testy = test.labels(idx,:);
    
    % normalise features
    %trainX_normalised = gradient_descent_preprocessing(trainX);
    trainX_normalised   = trainX;
    
    % build the objective function together with parameters for gradient descent
    objFct = softmaxcost_builder(trainX_normalised, trainy);
    startPt = randn(size(trainX, 2), 10);
    options.momentum = 0.5;
    options.timeStep = 1e-4;
    options.debugMode = 1;
    options.numIters = 1500;
    
    weights = gradient_descent(startPt, objFct, options);
    
    predictions = softmax(testX*weights);
    [prob, class_pred] = max(predictions, [], 2);
    
    error(i) = (sum((class_pred-1) ~=testy))*1/length(testy)*100;
end
figure;
boxplot(error);
hy = ylabel('% of error');
hx = xlabel('');
title('Softmax Regression Test Error');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir',...
'out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on
clear;
close all;

load data/train.mat;
load data/test.mat;

X = train.images(1:50000,:);
y = train.labels(1:50000,:);

validX = train.images(50001:end,:);
validy = train.labels(50001:end,:);

testX = test.images;
testy = test.labels;

%X = gradient_descent_preprocessing(X);

nn.batchSize = 100;  
nn.timeStep = 0.01;  
nn.momentum = 0.4; 
nn.epochs = 100; 
nn.lambda = 0;
nn.dropOut = 0;
nn = nn_builder(X, [300 300] , 10, 'logistic', nn);
tic
nn = nn_train(nn, X, y, validX, validy, 1, 0);
toc
[~, my_error] = nn_test(nn, testX, testy);


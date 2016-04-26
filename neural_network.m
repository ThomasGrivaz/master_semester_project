clear;
close all;

load data/train.mat;
load data/test.mat;

X = train.images(1:5000,:);
y = train.labels(1:5000,:);

validX = train.images(5001:10000,:);
validy = train.labels(5001:10000,:);

testX = test.images(1:1000,:);
testy = test.labels(1:1000,:);

nn.batchSize = 20;
nn.timeStep = 0.0001;
nn.momentum = 0.2;
nn.epochs = 100;


% normalise the features
X = gradient_descent_preprocessing(X);
nn = nn_builder(X, 50, 10, 'logistic', nn);
nn = nn_train(nn, X, y, validX, validy);
[labels, error] = nn_test(nn, testX, testy);


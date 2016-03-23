clear;
close all;

load data/train.mat;

% we keep only the 100 first samples to test
X = train.images(1:100,:);
y = train.labels(1:100,:);

% normalise the features
X = gradient_descent_preprocessing(X); 
nInputs = size(X,1);
nn = nn_builder(X, [100 100], 10, 'logistic', 'softmax');
nn = nn_fwd(nn, X);
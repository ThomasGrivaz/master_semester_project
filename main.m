%% This file is the main script to perform gradient descent



%% Data preprocessing

clear;
close all;

load data/train.mat;

% we keep only the 100 first samples to test
X = train.images(1:1000,:);
y = train.labels(1:1000,:);

% normalise the features
% X = gradient_descent_preprocessing(X); 
% features are already normalised, sparse matrix leads to NaN

%% Gradient descent
objFct = objective_fct_builder('l2', X, y);
[n, p] = size(X);
startPt = zeros(p, 1);
optArgs = struct('debugMode', 0);

weights = gradient_descent(startPt, objFct, optArgs);

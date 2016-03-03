%% This file is the main script to perform gradient descent



%% Data preprocessing

clear all;
close all;

load data/train.mat;

% we keep only the 100 first samples to test
X = train.images(1:100,:);
y = train.labels(1:100,:);

% normalise the features
%X = gradient_descent_preprocessing(X); 
% features are already normalised, sparse matrix leads to NaN

%% Gradient descent
objFct = objective_fct_builder('MSE', X, y);
[n, p] = size(X);
startPt = zeros(p, 1);

weights = gradient_descent(startPt, objFct);

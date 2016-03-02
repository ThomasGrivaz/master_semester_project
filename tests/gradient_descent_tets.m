clear all;
close all;

load data/train.mat;

X = train.images(1:100,:);
y = train.labels(1:100,:);
function [labels, missclass_rate] = nn_test(net, X, y)
%NN_TEST Test accuracy of neural network
%   
% Input parameters
%  net           : neural network previously trained
%  X             : test data matrix
%  y             : test labels vector
%
% Output parameters
% labels         : predicted labels
% missclass_rate : missclassification rate

% do a forward pass on test set, then retain max prob of last layer for
% labelling
net = nn_fwd(net, X, y);
[~, labels] = max(net.a{net.nLayers}, [], 2);

% allow classes to be labelled 0
labels = labels - 1;

% compute missclassification rate
err = (labels ~= y);
missclass_rate = (sum(err) / size(X,1))*100;


end


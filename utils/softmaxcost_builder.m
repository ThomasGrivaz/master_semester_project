function objFct = softmaxcost_builder(X, y)
%softmaxcost_builder this function returns the objective function for the
%softmax classification as well as its gradient
%   the softmax cost function forces the output to represent a probability
%   distribution across discrete alternatives, this function computes this
%   cost function given a dataset and a number of classes you want to
%   represent.
%
% Input parameters
%   X           :  design matrix
%   y           :  vector of labels
%   nbClasses  :  number of classes
%
% Output parameters
%   objFct.eval :  cost function
%   objFct.grad :  gradient

indicatorMatrix = full(sparse(1:length(y), y+1, 1));


objFct.eval = @(W) -sum(sum(indicatorMatrix.*log(softmax(X*W))));
objFct.grad = @(W) X'*(softmax(X*W)-indicatorMatrix);



end


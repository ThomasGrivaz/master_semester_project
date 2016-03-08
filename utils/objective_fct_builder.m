function fct = objective_fct_builder(objFct, X, y)
%objective_fct_builder This function returns a structure composed of a
%gradient and corresponding objective function depending on the input
%
% Input parameters 
%   objFct : the objective function to choose
%   X      : the design matrix
%   y      : the vector of labels
%
% Output parameters
%   fct    : a structure composed of a gradient field and a value field
switch objFct
    case 'l2'
        fct.grad = @(W) (1/length(y))*2*X'*(X*W - y);
        fct.eval = @(W) norm(X*W-y, 'fro')^2;
    case 'softmax'
        warning('not implemented yet')
    otherwise
        warning('Unexpected input. Choose either "l2" or "softmax"')
end
end


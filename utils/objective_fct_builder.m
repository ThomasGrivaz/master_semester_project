function fct = objective_fct_builder(objFct, X, y)
%objective_fct_builder This function returns a structure composed of a
%gradient and corresponding objective function depending on the input
%   Given an objective function to choose, the design matrix X and the
%   vector of labels y, the function returns a structure fct composed of a
%   .grad field used for the gradient and a .eval field used to evalue the
%   objective function
switch objFct
    case 'MSE'
        fct.grad = @(W) (1/length(y))*2*X'*(X*W - y);
        fct.eval = @(W) norm(X*W-y, 'fro')^2;
    case 'softmax'
        warning('not implemented yet')
    otherwise
        warning('Unexpected input. Choose either "MSE" or "softmax"')
end
end


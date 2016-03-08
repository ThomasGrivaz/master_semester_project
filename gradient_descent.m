function [w, stats] = gradient_descent(startPt, objFct, optArgs)
%gradient_descent.m Compute the gradient descent of a given objective
%    function from a specified starting point.
%
% Input parameters
%   startPt : starting point
%   objFct  : objective function to minimize
%   optArgs : structure of optional inputs
%
% Output parameters
%   w       : weight vector
%   stats   : structure containing statistics

% check if the correct number of inputs is entered
if nargin < 3
    optArgs = struct;
end

% initialise optional inputs
if ~isfield(optArgs, 'timeStep'), optArgs.timeStep = 0.001; end
if ~isfield(optArgs, 'numIters'), optArgs.numIters = 1000; end
if ~isfield(optArgs, 'debugMode'), optArgs.debugMode = 0; end

w = startPt;
costs = zeros(optArgs.numIters, 1);

% gradient descent
for k=1:optArgs.numIters
    
    % evalute gradient
    currentGradient= objFct.grad(w);
    
    if (optArgs.debugMode == 1);
        fprintf('Value of objective function: %f,', objFct.eval(w));
        fprintf('  Norm of current weight vector: %f \n', norm(w));
    end
    
    % update step
    w = w - optArgs.timeStep*currentGradient;
    
    % check if there is no divergence
    if ~isfinite(w)
        display(['Number of iterations: ' num2str(k)])
        error('w is inf or NaN')
    end
    
    if norm(currentGradient) < 1e-5;
        break;end;
    
    costs(k) = objFct.eval(w);
end

% gather statistics
stats.numIters = k;
stats.weightVectorNorm = norm(w);
stats.objFctValue = objFct.eval(w);

% plot evolution of cost function
if (optArgs.debugMode == 1);
    plot(costs);
    xlabel('number of iterations');
    ylabel('cost function');
end
end


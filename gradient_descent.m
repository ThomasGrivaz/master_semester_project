function [w, stats] = gradient_descent(startPt, objFct, varargin)
%gradient_descent.m Compute the gradient descent
%   Compute the gradient descent of a given objective function from a
%   specified starting point. The user can choose to manually set a time
%   step and the number of iterations 

% check if the correct number of inputs is entered
numvarargs = length(varargin);
if numvarargs > 2
    error('myfuns:gradient_descent:TooManyInputs', ...
        'requires at most 2 optional inputs');
end

% set default values for the time step and number of iterations if not
% specified
optargs = {0.001, 1000};

% overwrite values if specified
optargs(1:numvarargs) = varargin;

% initialise values
ALPHA = optargs{1};
MAX_ITERATIONS = optargs{2};
w = startPt;

% gradient descent
for k=1:MAX_ITERATIONS
    
    % evalute gradient
    currentGradient= objFct.grad(w);  
    fprintf('Value of objective function: %f,', objFct.eval(w));
    fprintf('  Norm of current weight vector: %f \n', norm(w));
    
    % update step
    w = w - ALPHA*currentGradient;
    
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
plot(costs);
xlabel('number of iterations');
ylabel('cost function');
end


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
%
% Options
%   optArgs.timeStep  : time step for each iteration of steepest descent
%   optArgs.numIters  : number of iterations
%   optArgs.debugMode : show some intel on the run such as the value of the
%   objective function, the norm of the weight vector and it outputs a plot
%   of the evolution of the cost function at the end.
%   optArgs.momentum  : momentum, if this is zero, we use steepest descent
%   optArgs.adaptive  : enables adaptive learning rate

% check if the correct number of inputs is entered
if nargin < 3
    optArgs = struct;
end

% initialise optional inputs
if ~isfield(optArgs, 'timeStep'), optArgs.timeStep = 0.001; end
if ~isfield(optArgs, 'numIters'), optArgs.numIters = 500; end
if ~isfield(optArgs, 'debugMode'), optArgs.debugMode = 0; end
if ~isfield(optArgs, 'momentum'), optArgs.momentum = 0; end
if ~isfield(optArgs, 'adaptive'), optArgs.adaptive = 0; end

w = startPt;
mu = optArgs.momentum;
alpha = optArgs.timeStep;
costs = zeros(optArgs.numIters, 1);
delta_w_old = zeros(size(w));

% gradient descent
for k=1:optArgs.numIters
    w_old = w;
    % evalute gradient
    currentGradient= objFct.grad(w_old);
    
    if (optArgs.debugMode == 1);
        fprintf('Value of objective function: %f,', objFct.eval(w_old));
        fprintf('  Norm of current weight vector: %f \n', norm(w_old));
    end
    
    % update step
    delta_w = -alpha*(1-mu)*currentGradient + mu*delta_w_old;
    w = delta_w + w_old;
    delta_w_old = delta_w;
    
    % update learning rate in case of adaptive learning rate
    if (optArgs.adaptive ==1)
        if objFct.eval(w_old) - objFct.eval(w) > 1e-4
            alpha = alpha*1.2;
        elseif objFct.eval(w_old) - objFct.eval(w) < 0
            alpha = alpha*0.5;
        end
    end
    
    
    % check if there is no divergence
    if ~isfinite(w) | ~isfinite(objFct.eval(w))
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
    hx= xlabel('number of iterations');
    hy = ylabel('cost function');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir',...
'out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
end
end


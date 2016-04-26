function net = nn_builder(X, structure, nOutput, actFct, net)
%NN_BUILDER Creates a neural network (MLP)
%
% Input parameters:
%   X            : data matrix
%   structure    : structure array of the neural network, i.e. the number of layers
%   with the number of neurons for each layer e.g. [h1 h2 h3] is a 3 layers
%   net with h1 hidden units for the first layer, h2 hidden units for the
%   second one and h3 hidden units for the third one.
%   nOutput      : number of outputs (classes)
%   actFct       : activation function of hidden units
%
% Output parameters:
%   net          : updated neural network
%
% net attributes:
%  actFct        : activation function
%  outputFct     : output function
%  lossFct       : loss function
%  w             : weights
%  delta_w_old   : difference of weights (see nn_gradient_step.m)
%  nLayers       : number of layers
%  timeStep      : time step for gradient descent
%  momentum      : momentum for gradient descent
%  batchSize     : size of batch to train
%  epochs        : number of epochs

%% initialize optional parameters

if nargin < 5
    net = struct;
end

if ~isfield(net, 'timeStep'), net.timeStep = 0.01; end
if ~isfield(net, 'momentum'), net.momentum = 0.5; end
if ~isfield(net, 'batchSize'), net.batchSize = 50; end
if ~isfield(net, 'epochs'), net.epochs = 50; end

%% initialize activation function

switch actFct
    case 'logistic'
        net.act = 'logistic';
        net.actFct = @(x) 1./(1 + exp(-x));
    case 'tanh'
        net.act = 'tanh';
        net.actFct = @(x) (exp(x) - exp(-x))/(exp(x) + exp(-x));
    otherwise
        warning('Choose either "logistic" or "tanh" as activation funcion');
end

%% initialize output function and loss function
if(nOutput < 2)
    warning('You need at least 2 classes');
elseif(nOutput == 2)
    % choose logistic if binary classification
    net.out = 'bin';
    net.outputFct = @(y) 1./(1 + exp(-y));
    net.lossFct = @(t,y) 1/2 * sum(sum((t-y).^2)) / size(t,1);
    
else
    % choose softmax if multi-class classification
    net.out = 'soft';
    net.outputFct = @(y) softmax(y);
    net.lossFct = @(t,y) -sum(sum(t.*log(y))) / size(t,1);
end
%%

% you need the dimensions of X for the normalisation factor and the number
% of columns of W{1}
[nInput, nFeatures] = size(X);

%% initialize weights and biases, also weight difference for momentum
net.nLayers = size(structure, 2) + 1;

% first layer is h1 * nFeatures, add bias
net.w{1} = randn(structure(1), nFeatures+1) * sqrt(2/nInput);
net.delta_w_old{1} = zeros(size(net.w{1}));

% layer i is h_i * h_i-1
for i= 2:net.nLayers-1
    net.w{i} = randn(structure(i), structure(i-1)+1) * sqrt(2/nInput);
    net.delta_w_old{i} = zeros(size(net.w{i}));
end

% last layer is nClasses * h_n
net.w{net.nLayers} = randn(nOutput, structure(net.nLayers-1)+1) * sqrt(2/nInput);
net.delta_w_old{net.nLayers} = zeros(size(net.w{net.nLayers}));
end


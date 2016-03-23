function net = nn_builder(X, structure, nOutput, actFct, outptFct)
%NN_BUILDER Creates a neural network (MLP)
% Input parameters:
%   nInput    : number of features
%   structure : structure array of the neural network, i.e. the number of layers
%   with the number of neurons for each layer e.g. [h1 h2 h3] is a 3 layers
%   net with h1 hidden units for the first layer, h2 hidden units for the
%   second one and h3 hidden units for the third one.
%   nOutput   : number of outputs
%   actFct    : activation function of hidden units
%   outptFct  : output function e.g. logistic or softmax

% initialise activation function
switch actFct
    case 'logistic'
        activation = @(x) 1./(1 + exp(-x));
    case 'tanh'
        activation = (@(x) exp(x) - exp(-x))/(exp(x) + exp(-x));
    otherwise
        warning('Choose either "logistic" or "tanh" as activation funcion');
end

switch outptFct
    case 'logistic'
        output = @(x) 1./(1 + exp(-x));
    case 'softmax'
        output = @(x) softmax(x);
    otherwise
        warning('Output is either "logistic" or "softmax"');
end
net.actFct = activation;
net.outptFct = output;
[nInput, nFeatures] = size(X);

% initialize weights and biases
nLayers = size(structure, 2);
net.nLayers = nLayers;
net.w{1} = randn(structure(1), nFeatures) * sqrt(2/nInput);
net.b{1} = zeros(structure(1), 1);
for i= 2:nLayers-1
    net.w{i} = randn(structure(i), structure(i-1)) * sqrt(2/nInput);
    net.b{i} = zeros(structure(i), 1);
end
net.w{nLayers} = randn(nOutput, structure(nLayers)) * sqrt(2/nInput);
end


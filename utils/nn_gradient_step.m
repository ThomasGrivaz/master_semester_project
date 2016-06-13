function net = nn_gradient_step(net)
%NN_GRADIENT_STEP Update the weights of the network according to gradient
%descent.
%
% Input Parameters:
%  net : neural network
%
% Output parameters:
%  net : neural network with updated weights

% for each layer, update weight
alpha = net.timeStep;
mu = net.momentum;
lambda = net.lambda;
for i = 1 : net.nLayers
    delta_w = -alpha*net.dW{i} + mu*net.delta_w_old{i};
   net.w{i} = delta_w + net.w{i} -alpha*lambda*[zeros(size(net.w{i},1), 1), net.w{i}(:,2:end)];
   net.delta_w_old{i} = delta_w;
    
end


end


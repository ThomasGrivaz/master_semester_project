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
for i = 1 : net.nLayers
    delta_w = -net.timeStep*(1-net.momentum)*net.dW{i} + net.momentum*...
        net.delta_w_old{i};
   net.w{i} = delta_w + net.w{i};
   net.delta_w_old{i} = delta_w;
    
end


end


function net = nn_bwd(net, X)
%NN_BWD Backward propagation through the neural network
%
% Input parameters:
%  net : neural network
%  X   : input matrix
%
% Output parameters:
%  net : neural network with updated weights
n = net.nLayers;

% compute derivatives of error wrt last layer of activations
switch(net.out)
    case 'bin'
        net.r{n} = - net.e .* (net.a{n} .* (1 - net.a{n}));
    case 'soft'
        net.r{n} = -net.e;
end

% backpropagate errors wrt activations: r_i = diag(g'(a_i)) * W_i+1 * r_i+1
for i = (n-1) : -1 : 1
    
    % compute g'(a_i)
    switch(net.act)
        case 'logistic'
            delta = net.h{i} .* (1 - net.h{i});
        case 'tanh'
            delta = sech(net.h{i}).^2;
    end
    if (i+1) == n
        net.r{i} = (net.r{i+1} * net.w{i+1}) .* delta;
    else
        net.r{i} = (net.r{i+1}(:,2:end) * net.w{i+1}) .* delta;
    end
    
end

% use derivative to compute adjustements to be made to weights
net.dW{1} = net.r{1}(:,2:end)' * [ones(size(X,1), 1),X];
for i =1 : (n-1)
    if (i+1) == n
        net.dW{i+1} = net.r{i+1}' * net.h{i};
    else
        net.dW{i+1} = net.r{i+1}(:,2:end)' * net.h{i};
    end
end

end


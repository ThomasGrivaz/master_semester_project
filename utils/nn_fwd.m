function net = nn_fwd(net, X)
%NN_FWD Forward pass through the neural network
%   Detailed explanation goes here
g = net.actFct;

net.a{1} = X*net.w{1}' + ones(1, size(X,1))*net.b{1};
net.h{1} = g(net.a{1});
n = net.nLayers;

% update activation and hidden units
for i = 2 : n -1
    net.a{i} = net.h{i-1}*net.w{i}' + ones(1, size(net.w{i}, 1))*net.b{i};
    net.h{i} = g(net.a{i});
end
net.a{n} = net.outptFct(net.h{n-1}*net.w{n}');
[~, net.a{n}] = max(net.a{n}, [], 2);
end


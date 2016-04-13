function net = nn_fwd(net, X, y)
%NN_FWD Forward pass through the neural network
%
% Input parameters:
%  net : neural network
%  X   : design matrix
%  y   : labels vector
%
% Output parameters:
%  net : updated neural network


g = net.actFct;
n = net.nLayers;
X = [ones(size(X,1), 1), X];

% update first layer
net.a{1} = X*net.w{1}';
net.h{1} = g(net.a{1});


% update activation and hidden units
for i = 2 : n -1
    net.a{i} = net.h{i-1}*net.w{i}';
    net.h{i} = g(net.a{i});
end

% update last layer
net.a{n} = net.outputFct(net.h{n-1}*net.w{n}');

% compute error
y = full(sparse(1:length(y), y+1, 1));
net.loss = net.lossFct(y, net.a{n});
net.e = y - net.a{n};

% take max probability and assign it to respective class
%[~, net.a{n}] = max(net.a{n}, [], 2);

% allow classes to be labelled 0
%net.a{n} = net.a{n} - 1;


end


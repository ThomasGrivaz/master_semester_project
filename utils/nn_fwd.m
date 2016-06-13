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
m = size(X,1);
X = [ones(m, 1), X];

% update first layer
net.a{1} = X*net.w{1}';
net.z{1} = g(net.a{1});

net.z{1} = [ones(m,1), net.z{1}];

if (net.dropOut > 0)
    net.dropout_mask{1} = binornd(1, 1-net.dropOut, size(net.z{1}));
    net.z{1} = net.z{1}.* net.dropout_mask{1};
end



% update activation, bias and hidden units
for i = 2 : n - 1
    net.a{i} = net.z{i-1}*net.w{i}';
    
    net.a{i} = [ones(m, 1), net.a{i}];
    net.z{i} = g(net.a{i});
    
    if (net.dropOut > 0)
        net.dropout_mask{i} = binornd(1, 1-net.dropOut, size(net.z{i}));
        net.z{i} = net.z{i}.*net.dropout_mask{i};
    end
    
end

% update last layer
net.a{n} = net.outputFct(net.z{n-1}*net.w{n}');

% compute error
nClasses = size(net.a{n},2);
y_matrix = zeros(m, nClasses);

% preprocess vector of labels to compute errors (transform into binary
% matrix)
for i = 1 : nClasses
    y_matrix(:,i) = 1*((y + 1) == i);
end

%NB: this method does not work if a batch has not all classes represented,
%the number of columns of y wont match with net.a{n}
%y_matrix = full(sparse(1:length(y), y+1, 1));

net.loss = net.lossFct(y_matrix, net.a{n});
net.e = y_matrix - net.a{n};



end


clear
load data/train.mat;
X = train.images(1:1000,:);
y = train.labels(1:1000,:);
X1 = X(1:500,:);
X2 = X(501:end,:);
y1 = y(1:500,:);
y2 = y(501:end,:);

net = nn_builder(X, [100 100], 10, 'logistic');

net = nn_fwd(net, X, y);
net1 = nn_fwd(net,X1, y1);
net2 = nn_fwd(net, X2, y2);

net = nn_bwd(net, X);
net1 = nn_bwd(net1, X1);
net2 = nn_bwd(net2, X2);
for i = 1 : net.nLayers
    discrepancy{i} = net1.dW{i} + net2.dW{i}- net.dW{i};
    layer_discrepancy = norm(discrepancy{i},'fro');
    assert( layer_discrepancy < 20, 'error: derivatives not equals');
end
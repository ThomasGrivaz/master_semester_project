function [net, trainLoss, validLoss] = nn_train(net, X_train, y_train, X_valid, y_valid)
%NN_TRAIN Trains a neural network
%   Detailed explanation goes here
batch_size = net.batchSize;
batches = size(X_train,1) / batch_size;

%trainLoss = zeros(net.epochs, 1);
%validLoss = zeros(net.epochs, 1);
NbIterations = 0;
for i= 1 : net.epochs
    
    rng('shuffle')
    % shuffle data
    indices = randperm(size(X_train,1))';
    X_train = X_train(indices,:);
    y_train = y_train(indices,:);
    X_valid = X_valid(indices,:);
    y_valid = y_valid(indices,:);
    
    for j = 1 : batches
        
        batch_X_train = X_train((j - 1) * batch_size + 1 : j * batch_size,:);
        batch_y_train = y_train((j - 1) * batch_size + 1 : j * batch_size,:);
        
        net = nn_fwd(net, batch_X_train, batch_y_train);
        net = nn_bwd(net, batch_X_train);
        net = nn_gradient_step(net);
        
    end
    
    [~,trainLoss(i)] = nn_test(net, X_train, y_train);
    [~,validLoss(i)] = nn_test(net, X_valid, y_valid);
    
    fprintf('epoch: %d, train error: %.2f, validation error: %.2f\n',...
        i, trainLoss(i), validLoss(i));
    NbIterations(i) = i;
    if ( i > 1 && validLoss(i) - validLoss(i-1) > 2*10e-2)
        %break
    end
    
end

plot(NbIterations, trainLoss, 'b', NbIterations, validLoss, 'r');
hx = xlabel('epoch');
hy = ylabel('Error');
legend('Train error', 'Validation error', 'Location', 'northeast');

% the following code makes the plot look nice and increase font size etc.
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;


end


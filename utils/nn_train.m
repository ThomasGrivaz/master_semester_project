function [net, trainLoss, validLoss] = nn_train(net, X_train, y_train, X_valid, y_valid,plot_flag, early_stopping_flag)
%NN_TRAIN Trains a neural network
%   Detailed explanation goes here
batch_size = net.batchSize;
nBatches = size(X_train,1) / batch_size;

%trainLoss = zeros(net.epochs, 1);
%validLoss = zeros(net.epochs, 1);
NbIterations = 0;
for i= 1 : net.epochs
    
    rng('shuffle')
    % shuffle data
    indicesTrain = randperm(size(X_train,1))';
    indicesValid = randperm(size(X_valid,1))';
    X_train = X_train(indicesTrain,:);
    y_train = y_train(indicesTrain,:);
    X_valid = X_valid(indicesValid,:);
    y_valid = y_valid(indicesValid,:);
    
    for j = 1 : nBatches
        
        batch_X_train = X_train((j - 1) * batch_size + 1 : j * batch_size,:);
        batch_y_train = y_train((j - 1) * batch_size + 1 : j * batch_size,:);
        
        net = nn_fwd(net, batch_X_train, batch_y_train);
        net = nn_bwd(net, batch_X_train);
        if ~(isfinite(net.dW{1})), error('derivatives not finite'); end
        net = nn_gradient_step(net);
        
    end
    
    % compute loss for both sets
    net_tmp = nn_fwd(net, X_train, y_train);
    trainLoss(i) = net_tmp.loss;
    
    net_tmp = nn_fwd(net, X_valid, y_valid);
    validLoss(i) = net_tmp.loss;

    % or choose this if you want to plot accuracy
%     [~ ,trainLoss(i)] = nn_test(net, X_train, y_train);
%     [~, validLoss(i)] = nn_test(net, X_valid, y_valid);
%     trainLoss(i) = 100 - trainLoss(i);
%     validLoss(i) = 100 - validLoss(i);
    
    
    
    fprintf('epoch: %d, train error: %.4f, validation error: %.4f\n',...
        i, trainLoss(i), validLoss(i));
    NbIterations(i) = i;
    if(early_stopping_flag == 1)
        if ( i > 3 && validLoss(i) - validLoss(i-1) >= 0 && validLoss(i-1) -...
                validLoss(i-2) >= 0 && validLoss(i-2) - validLoss(i-3) >= 0) || ...
                (i > 10 && abs(validLoss(i-1) - validLoss(i)) <= 3e-3)
            break
        end
    end
    
end
if(plot_flag==1)
    plot(NbIterations, trainLoss, 'b', NbIterations, validLoss, 'r');
    hx = xlabel('epoch');
    hy = ylabel('error %');
    legend('Train error', 'Validation error', 'Location', 'northeast');
    
    % the following code makes the plot looks nice and increase font size etc.
    set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
    set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
    grid on;
end

end


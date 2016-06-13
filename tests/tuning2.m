clear;
close all;

load data/train.mat;
load data/test.mat;

X = train.images;
y = train.labels;


testX = test.images;
testy = test.labels;

K = 4;

param = linspace(0, 0.8, 9);
test_errors = zeros(K,length(param));
my_error = zeros(1,length(param));
indices = crossvalind('Kfold', 60000, 5);
for k=1:K
    train_idx = find(indices~=k);
    valid_idx = find(indices==k);
    trainX = X(train_idx,:);
    trainy = y(train_idx,:);
    validX = X(valid_idx,:);
    validy = y(valid_idx,:);
    fprintf('k: %d \n',k);
    
    for i=1:length(param)
        nn.batchSize = 100;
        nn.timeStep = 0.01;
        nn.momentum = param(i);
        nn.epochs = 10;
        nn.lambda = 0;
        nn.dropOut = 0;
        %neurons = param(i);
        %archi = 150*ones(1,param(i));
        archi = [200 200];
        nn = nn_builder(trainX, archi, 10, 'logistic', nn);
        nn = nn_train(nn, trainX, trainy, validX, validy, 0, 0);
        [~, my_error(i)] = nn_test(nn, testX, testy);
    end
    test_errors(k,:) = my_error;
end
%% Plots
mean_curve = mean(test_errors);
figure;
plot(param, test_errors(1,:), 'b', param, test_errors(2,:), 'g',...
    param, test_errors(3,:), 'r', param, test_errors(4,:), 'm');
hold on
plot(param, mean_curve, 'k','LineWidth', 5);
hy = ylabel('Classification Error %');
hx = xlabel('Momentum');
%title('Test Error');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir',...
    'out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on
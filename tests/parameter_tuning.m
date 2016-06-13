clear;
close all;

load data/train.mat;
load data/test.mat;

X = train.images;
y = train.labels;


testX = test.images;
testy = test.labels;

K = 4;

%param = [10,20,25,40,50,100,125,200,250,400]; %batch
%param = [50, 100, 150, 200, 250, 300, 350]; %hidden units
%param = [1, 2, 3]; %layers
%param = [0, 0.1, 0.2, 0.3, 0.4]; %dropout
param = logspace(-3, -1, 4);
test_errors = zeros(K,length(param));
my_error = zeros(K,1);
for i=1:length(param)
    fprintf('value of param: %d, iteration: %d / %d \n', param(i), i, length(param));
    nn.batchSize = 100;
    nn.timeStep = 0.01;
    nn.momentum = 0.4;
    nn.epochs = 10;
    nn.lambda = param(i);
    nn.dropOut = 0;
    neurons = 300;
    %archi = 150*ones(1,param(i));
    archi = [300 300];
    indices = crossvalind('Kfold', 60000, 5);
    for k=1:K
        train_idx = find(indices~=k);
        valid_idx = find(indices==k);
        trainX = X(train_idx,:);
        trainy = y(train_idx,:);
        validX = X(valid_idx,:);
        validy = y(valid_idx,:);
        nn = nn_builder(trainX, archi, 10, 'logistic', nn);
        nn = nn_train(nn, trainX, trainy, validX, validy, 0, 0);
        [~, my_error(k)] = nn_test(nn, testX, testy);
    end
    test_errors(:,i) = my_error;
end
%% Plots
figure;
boxplot(test_errors, 'labels', param);
hy = ylabel('Classification Error %');
hx = xlabel('Dropout fraction');
%title('Test Error');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir',...
'out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on
%% Convergence Graphics
%This script runs gradient descent and outputs a nice plot to help
%visualising how convergence goes depending on the starting point and the
%time step. The data and some portions of code come from the PCML class of
%2014/2015.
%http://edu.epfl.ch/coursebook/fr/pattern-classification-and-machine-learning-CS-433
%%

% Load data and convert it to the metrics system
load('height_weight_gender.mat');
height = height * 0.025;
weight = weight * 0.454;

% normalize features (store the mean and variance)
x = height;
meanX = mean(x);
x = x - meanX;
stdX = std(x);
x = x./stdX;

% Form (y,tX) to get regression data in matrix form
y = weight;
N = length(y);
tX = [ones(N,1) x(:)];

% generate a grid of values for beta0 and beta1
beta0 = [-150:1:250];
beta1 = [-200:1:200];
L = zeros(length(beta0), length(beta1));
L_lipschitz = zeros(length(beta0), length(beta1));

% anonymous function to compute MSE
computeCost = @(y, tX, weights) (y-tX*weights)'*(y-tX*weights)/(2*length(y));

% anonymous function to compute the gradient of MSE
computeGradient = @(y, tX, weights) (-1/length(y))*tX'*(y-tX*weights);

% grid search for beta0, beta1
for i = 1:length(beta0)
    for j =1:length(beta1)
        L(i,j) = computeCost(y, tX, [beta0(i); beta1(j)]);
    end
end

% compute minimum value of L and also beta0_star and beta1_star
[val, idx] = min(L(:));
[i_min,j_min] = ind2sub(size(L), idx);
beta0_star = beta0(i_min);
beta1_star = beta1(j_min);
L_star = L(i_min, j_min);


% contour plot
figure(1)
clf;
subplot(221);
contourf(beta0, beta1, L', 20); colorbar;
hx = xlabel('\beta_0');
hy = ylabel('\beta_1');
hold on
plot(beta0_star, beta1_star, 'h',...% put a marker at the minimum
    'markersize', 7, 'markerfacecolor',[1 1 1]...
    ,'markeredgecolor',[1 1 1], 'linewidth',1);
set(gca, 'fontsize', 14);

subplot(223);
contourf(beta0, beta1, L', 20); colorbar;
hx = xlabel('\beta_0');
hy = ylabel('\beta_1');
hold on
plot(beta0_star, beta1_star, 'h',...% put a marker at the minimum
    'markersize', 7, 'markerfacecolor',[1 1 1]...
    ,'markeredgecolor',[1 1 1], 'linewidth',1);
set(gca, 'fontsize', 14);


% algorithm parametes
maxIters = 50;
alpha = 2; %divergence for 1.5 /1.6
lipschitz_cst = 2;
converged = 0;

% initialize
beta = [-100; 150];
beta_lipschitz = [-100;150];

for k = 1:maxIters
    % compute gradient
    g = computeGradient(y,tX,beta);
    g_lipschitz = computeGradient(y, tX, beta_lipschitz);
    
    % compute cost function
    L = computeCost(y, tX, beta);
    L_lipschitz = computeCost(y, tX, beta_lipschitz);
    
    % update step
    beta = beta - alpha.*g;
    beta_lipschitz = beta_lipschitz - 1/lipschitz_cst.*g_lipschitz;
    
    
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;
    
    % print
   % fprintf('%.2f  %.2f %.2f\n', L, beta(1), beta(2));
    
    % Overlay on the contour plot
    % For this to work you first have to run grid Search
    subplot(221);
    plot(beta(1), beta(2), 'o', 'color', [1 0 0], 'markersize', 7);
    pause(.5) % wait half a second
    
    subplot(223);
    plot(beta_lipschitz(1), beta_lipschitz(2), 'o', 'color', [1 0 0], 'markersize', 7);
    pause(.5) % wait half a second
    
    % visualize function f on the data
    subplot(222);
    x = [1.2:.01:2]; % height from 1m to 2m
    x_normalized = (x - meanX)./stdX;
    f1 = beta(1) + beta(2).*x_normalized;
    plot(height, weight,'.');
    hold on;
    plot(x,f1,'r-');
    hx = xlabel('x');
    hy = ylabel('y');
    hold off;
    
    subplot(224);
    x = [1.2:.01:2]; % height from 1m to 2m
    x_normalized = (x - meanX)./stdX;
    f2 = beta_lipschitz(1) + beta_lipschitz(2).*x_normalized;
    plot(height, weight,'.');
    hold on;
    plot(x,f2,'r-');
    hx = xlabel('x');
    hy = ylabel('y');
    hold off;
end



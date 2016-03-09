function X_normalised = gradient_descent_preprocessing(X)
%gradient_descent_preprocessing Rescales the features to garanty better
%convergence for gradient descent
%
% Input parameters
%   X : data
%
% Output parameters
%   X_normalised : centered data


% extract mean and standard deviation for each column
mu = mean(X);
sigma = std(X);
sigma=max(std(X),eps);

X_normalised = bsxfun(@minus, X, mu);
X_normalised = bsxfun(@rdivide, X_normalised, sigma);

end


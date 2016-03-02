function X_normalised = gradient_descent_preprocessing(X)
%gradient_descent_preprocessing Rescales the features to garanty better
%convergence for gradient descent
%   We first extract the mean and the standard deviation of the data, then
% we center it so that the output has mean 0 and standard deviation 1


mu = mean(X);
sigma = std(X);

X_normalised = bsxfun(@minus, X, mu);
X_normalised = bsxfun(@rdivide, X, sigma);

end


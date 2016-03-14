function prob = softmax(x)
%SOFTMAX compute the softmax function
%   formula : Pr(y=i| X,W) = exp(X'*W_i) / sum_j(X'*w_j)

num = exp(x);
denom = sum(num);

prob = bsxfun(@rdivide, num, denom);


end


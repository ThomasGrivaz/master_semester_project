function Nb_err = gradient_descent_tets()

Nb_err = 0;
Nb_err = Nb_err + test_simple_gradient_descent_l2();


end


function err = test_simple_gradient_descent_l2()
N = 100;
M = 20;

X = randn(N,M);
W_gt = randn(M,1);

y = X * W_gt;


ftest.eval = @(W) norm(X*W-y,'fro')^2;
ftest.grad = @(W) 2*X'*(X*W-y);

optArgs.momentum = 0.5;
options.adaptive = 1;

Win = rand(M,1);

[W_opt1, stats1] = gradient_descent(Win,ftest, optArgs);
[W_opt2, stats2] = gradient_descent(Win,ftest);
[W_opt3, stats3] = gradient_descent(Win,ftest,options);
fprintf('number of iterations with momentum: %d\n', stats1.numIters);
fprintf('number of iterations without momentum: %d\n', stats2.numIters);
fprintf('number of iterations with adaptive learning rate: %d',...
    stats3.numIters);

err = (norm(W_gt - W_opt1,'fro') > 1e-5)+ (norm(W_gt - W_opt2,'fro') > 1e-5)...
    +(norm(W_gt - W_opt3,'fro') > 1e-5);

end

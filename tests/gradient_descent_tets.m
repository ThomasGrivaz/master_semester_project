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
ftest.gradlipshiptz = 2*norm(X)^2;

optArgs.timeStep = 1/ftest.gradlipshiptz

Win = rand(M,1);

[W_opt, stats1] = gradient_descent(Win,ftest, optArgs);
%[W_opt, stats2] = gradient_descent(Win,ftest);

err = (norm(W_gt - W_opt,'fro') > 1e-5);

end
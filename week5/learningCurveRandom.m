function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)

% Number of training examples
m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m
    for j = 1:50
        sel = randperm(m); sel = sel(1:i);
        [theta] = trainLinearReg(X(sel,:), y(sel), lambda);
        error_train(i) = error_train(i) + linearRegCostFunction(X(sel,:), y(sel), theta, 0);
        error_val(i) = error_val(i) + linearRegCostFunction(Xval(sel,:), yval(sel), theta, 0);
    end
    
    error_train(i) = error_train(i) / 50;
    error_val(i) = error_val(i) / 50;
end

end

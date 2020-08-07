function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = C_vec;
vec_size = size(C_vec, 1);
pred_error = zeros(vec_size^2, 1);

for i = 1:vec_size
    for j = 1:vec_size
        model= svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));
        predictions = svmPredict(model, Xval);
        pred_error(vec_size*(i-1)+j) = mean(double(predictions ~= yval));
    end
end
%disp(pred_error); fprintf(['size(pred_error): %f'], size(pred_error));
[~, min_error_index] = min(pred_error);
C = C_vec(ceil(min_error_index/vec_size));
sigma = sigma_vec(mod(min_error_index, vec_size));

% =========================================================================

end

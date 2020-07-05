function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


C = 1;
sigma = 0.3;
rand_C = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
rand_sigma = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
        
test_error = mean(double(predictions ~= yval));

for i=1:size(rand_C)
    for j=1:size(rand_C)
        
        model= svmTrain(X, y, rand_C(i), @(x1, x2) gaussianKernel(x1, x2, rand_sigma(j)));
        predictions = svmPredict(model, Xval);
        
        error = mean(double(predictions ~= yval));
        
        if error <= test_error
            C = rand_C(i);
            sigma = rand_sigma(j);
            test_error = error;
        end
    end
end

C
sigma






% =========================================================================

end

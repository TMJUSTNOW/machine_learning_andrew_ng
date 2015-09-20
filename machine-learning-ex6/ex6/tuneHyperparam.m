function [C, sigma] = tuneHyperparam()
%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

%  This is a different dataset that you can use to experiment with. Try
%  different values of C and sigma here.
% 

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Try different SVM Parameters here
accuracy = [];
for C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        % Train the SVM
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        pred = svmPredict(model, Xval);
        accuracy = [accuracy; [C sigma sum(yval == pred) / length(yval)]];
    end
end

[value index] = max(accuracy(:,3));
C       = accuracy(index, 1);
sigma   = accuracy(index, 2);

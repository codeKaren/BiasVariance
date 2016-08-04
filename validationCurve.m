function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

% loop through each value of lambda
for i = 1:length(lambda_vec)
	lambda = lambda_vec(i);
	% use trainLinearReg() to compute theta using current value of lambda and training set
	theta = trainLinearReg(X, y, lambda);
	% get the cost/error for the training set 
	% we pass in 0 for lambda because we don't need to regularize twice (theta already regularized)
	[costTrain, gradTrain] = linearRegCostFunction(X, y, theta, 0);
	% get the cost/error for the CV set 
	[costCV, gradCV] = linearRegCostFunction(Xval, yval, theta, 0);
	% initialize error_train and error_val with the computed costs
	error_train(i) = costTrain;
	error_val(i) = costCV;
end

% =========================================================================

end

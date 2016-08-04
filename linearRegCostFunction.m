function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% first, compute the cost function without the regularization term
% theta is n x 1 and X is m x n, so hypothesis is m x 1
hypothesis = X * theta;
costTerm = sum ((hypothesis - y) .^ 2) ./ (2.*m);
% compute the regularization term, but don't regularize theta(1)
regularizationTheta = theta;
regularizationTheta(1) = 0;
regularizationTerm = lambda./(2.*m).*sum(regularizationTheta .^ 2);
% compute the cost
J = costTerm + regularizationTerm;

% second, compute the gradient of the cost function
% compute the gradient term without regularization
% gradTerm must be n x 1 because gradRegularizationTerm is n x 1
% difference is m x 1
difference = hypothesis - y; 
% X' is n x m, so X' * difference is n x 1
gradTerm = (X' * difference)./m; 
% compute the regularization term, but don't regularize theta(1)
% use regularizationTheta from above since theta(1) is set to 0 there
gradRegularizationTerm = lambda./m.*(regularizationTheta);
% compute the gradient
grad = gradTerm + gradRegularizationTerm;

% =========================================================================

grad = grad(:);

end

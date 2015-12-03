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


r = 1 / (2 * m);
thetaNoFirstParam = theta;
thetaNoFirstParam(1) = 0;
regParam = (lambda / (2 * m)) * sum(thetaNoFirstParam .^ 2);

J = (r * sum( ((X * theta)-y) .^ 2 )) + regParam;


r2          = 1 / m;
regParam2   = ( (lambda / m) .* theta(2:end) );
theta_temp1 = (r2 .*  sum( ((X * theta)-y) .*  X(:,1) )); 

newX = X(:,2:(size(X)(2)));
theta_temp2 = (r2 .*  (newX' * ((X * theta)-y))) + regParam2;



% =========================================================================

grad =  [theta_temp1;theta_temp2];
grad = grad(:);

end

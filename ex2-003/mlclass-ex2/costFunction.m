function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

r = 1/m;
alpha = 0.01;
z = (X * theta);
Y1 = (y' * log(sigmoid(z)));
Y2 = ((1 - y)' * log(1 - sigmoid(z)));

J = r * sum(-Y1 - Y2);

theta_temp1 = r *  sum( (sigmoid(z)-y) .*  X(:,1) ); 
theta_temp2 = r *  sum( (sigmoid(z)-y) .*  X(:,2) );
theta_temp3 = r *  sum( (sigmoid(z)-y) .*  X(:,3) );

theta(1) = theta_temp1;
theta(2) = theta_temp2;
theta(3) = theta_temp3;
%fprintf("barry cost func %f \n", J);

grad = theta;

 






% =============================================================

end

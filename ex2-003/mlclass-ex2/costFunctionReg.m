function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

r2 = lambda/(2 * m);
RegParam = r2 * sum(theta(2:end) .^ 2); 



r = 1/m;

z = (X * theta);
Y1 = (y' * log(sigmoid(z)));
Y2 = ((1 - y)' * log(1 - sigmoid(z)));

J = (r * sum(-Y1 - Y2)) + RegParam;


theta_temp1 = (r .*  sum( (sigmoid(z)-y) .*  X(:,1) ));
regular =  ( (lambda / m) .* theta(2:end) );

newX = X(:,2:(size(X)(2)));

theta_temp2 = (r .*  (newX' * (sigmoid(z)-y))) + regular;
%theta = [theta_temp1, theta_temp2]; 
%theta_temp3 = (r *  sum( (sigmoid(z)-y) .*  X(:,3) )) + ( (lambda / m) * theta(3));

%theta(1) = theta_temp1;
%theta(2:(size(theta_temp2)(1)+1),:) = theta_temp2;

%size(theta(2:27,:))
% size(theta_temp2)
 
 %size(theta_temp2)
%size(( (lambda / m) * theta(3)))
%theta = theta_temp2;
%theta(2) = theta_temp2;
%theta(3) = theta_temp3;


%fprintf("barry test --");
grad = [theta_temp1;theta_temp2];
% =============================================================

end

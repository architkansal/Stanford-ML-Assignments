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

	h_x = X*theta;
	h_x = sigmoid(h_x);
	[m,n] = size(y);

	theta2 = theta;
	theta2(1) = 0;

	J = -1*(1/m)*((y')*log(h_x) + ((ones(size(y))-y)')*log(ones(size(h_x))-h_x)) + (lambda/(2*m))*sum((theta2.*theta2)); 
	theta3 = theta;
	theta3(1) = 0;

	grad = (1/m)*(X'*(h_x-y)) + (lambda/m)*(theta3);



% =============================================================

end

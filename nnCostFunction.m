function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%% Part 1: Feedforward & Backpropagation of Errors

X = [ones(m, 1), X];
for i = 1:m
    % extracting each example one by one into a column vector
    a1 = X(i, :)';
    % calulating the preactivation and activation values for 1st hidden layer
    z2 = Theta1 * a1;
    a2 = ones(hidden_layer_size + 1, 1);
    a2(2:end) = sigmoid(z2);
    % calulating the preactivation and activation values for the output layer
    z3 = Theta2 * a2;
    a3 = sigmoid(z3); % h(x) = a3
    % converting the label to a column vector to represent the target
    y_vec = zeros(num_labels, 1);
    y_vec(y(i)) = 1;
    % cost caluclation
    J = J + sum(-(y_vec) .* log(a3) -(1 - y_vec) .* log(1 - a3));
    
    % backpropagation of errors
    % error in output layer
    delta_3 = a3 - y_vec;
    % derivative of cost function J(theta) w.r.t parameters Theta2
    Theta2_grad = Theta2_grad + delta_3 * a2';
    % error in hidden layer
    delta_2 = (Theta2' * delta_3);
    delta_2 = delta_2(2:end) .* sigmoidGradient(z2); % getting rid of error in the bias term
    % derivative of cost function J(theta) w.r.t parameters Theta1
    Theta1_grad = Theta1_grad + delta_2 * a1';
end
J = J/m;

Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;

% converting the un-regularized cost function to regularized cost function
R = 0;

% computing cost from parameters of Layer1
for j = 1:hidden_layer_size
    for k = 2:input_layer_size + 1 % not regularizing the term corresponding to the bias
        R = R + Theta1(j, k)^2;
        if k > 1
            Theta1_grad(j, k) = Theta1_grad(j, k) + lambda/m * Theta1(j, k);
        end
    end
end

% computing cost from parameters of Layer2
for j = 1:num_labels
    for k = 2:hidden_layer_size + 1
        R = R + Theta2(j, k)^2;
        if k > 1
            Theta2_grad(j, k) = Theta2_grad(j, k) + lambda/m * Theta2(j, k);
        end
    end
end

R = lambda/(2 * m) * R;

% Total regularized cost
J = J + R;

% Unrolled gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10


%% Loading and Visualizing Data

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data1.mat');

% Split data into training, validation & test set
test_indices = [];
val_indices = [];
train_indices = [];
for i = 0:9
    start_index_test = i * 500 + 1;
    end_index_test = start_index_test + 99;
    start_index_val = end_index_test + 1;
    end_index_val = start_index_val + 79;
    start_index_train = end_index_val + 1;
    end_index_train = (i + 1) * 500;
    test_indices = [test_indices, start_index_test:end_index_test];
    val_indices = [val_indices, start_index_val:end_index_val];
    train_indices = [train_indices, start_index_train:end_index_train];
end

% test data
X_test = X(test_indices, :);
y_test = y(test_indices, :);

% validation data
% X_val = X(val_indices, :);
% y_val = y(val_indices, :);

% training data
% X = X(train_indices, :);
% y = y(train_indices, :);

% training the model using the training set & the validation set together
X = X([train_indices, val_indices], :);
y = y([train_indices, val_indices], :);

m = size(X, 1);

% Randomly selected 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('\nPress enter to continue.\n');
pause;

%% Initializing Pameters

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nPress enter to continue.\n');
pause;

%% Gradient Checking

fprintf(['\nChecking Backpropagation implementation using gradient check\n'...
        'i.e., Numerical Gradient & Analytic gradient comparision\n\n']);

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

fprintf('\nPress enter to continue.\n');
pause;

%% Training NN
%  We use an optimization function fmincg for optimizing our parameters
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 30);

lambda = 2;

% cost function
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% costFunction takes in only one argument (the neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Press enter to continue.\n');
pause;

%% Prediction and Accuracy
% Training
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% Validataion
% pred = predict(Theta1, Theta2, X_val);
% fprintf('\nValidation Set Accuracy: %f\n', mean(double(pred == y_val)) * 100);

% Test
pred = predict(Theta1, Theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

fprintf('Press enter to continue.\n');
pause;

%% Visualize Weights

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nPress enter to end.\n');
pause;


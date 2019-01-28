function g = sigmoid(z)
%SIGMOID Compute sigmoid functon

g = 1.0 ./ (1.0 + exp(-z));
end

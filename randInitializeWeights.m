function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections for each unit

W = zeros(L_out, 1 + L_in);

% The first column of W corresponds to the parameters for the bias units

epsilon_init = sqrt(6) / sqrt(L_in + L_out);
W = -epsilon_init + 2 * epsilon_init * rand(size(W));

end

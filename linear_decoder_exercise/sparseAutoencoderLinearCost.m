function [cost,grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

b1rep = repmat(b1,1,size(data,2));
b2rep = repmat(b2,1,size(data,2));

hidden1 = sigmoid( W1 * data + b1rep );
output  =  W2 * hidden1 + b2rep ;

sparse = mean(hidden1,2);

grad1 = output-data;
W2grad =  grad1*hidden1';
b2grad = sum(grad1,2);

grad2 =  ((W2'*grad1)+ beta.*repmat( ((1-sparsityParam)./(1-sparse) - sparsityParam./sparse) ,1, size(hidden1,2)) ).*hidden1.*(1-hidden1);
W1grad = grad2*data';
b1grad = sum(grad2,2);

W1grad = W1grad/size(data,2) + lambda*W1;
W2grad = W2grad/size(data,2) + lambda*W2;
b1grad = b1grad/size(data,2);
b2grad = b2grad/size(data,2);

spasePenalty = beta*sum( sparsityParam * log( sparsityParam./sparse) + (1-sparsityParam)*log( (1-sparsityParam)./(1-sparse)));
weightdecay = 0.5*lambda*( sum(sum( W1.*W1))+sum(sum(W2.*W2)) );
cost = (0.5*sum(sum((output-data).*(output-data))))/size(data,2) + spasePenalty + weightdecay;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

% test = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
% weight = getweight( test, 1, 1, 2, 2)
% getgradient( weight, 1, 1, 2, 2, size(test,1), size(test,2) )
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function weight = getweight( W, padw, padh, fw, fh )
    wnum = (size(W,1)-fw)/padw+1;
    hnum = (size(W,2)-fh)/padh+1;
    weight = zeros(fw*fh, wnum*hnum);
    for i=1:hnum
        for j=1:wnum 
            t = W( 1+padw*(j-1):1+padw*(j-1)+fw-1, 1+padh*(i-1):1+padh*(i-1)+fh-1);
            weight(:, (i-1)*wnum+j ) = reshape(t, fw*fh, 1 );
        end
    end
end

function gradient = getgradient( grad, padw, padh, fw, fh, row, col)
    gradient = zeros(row, col);
    wnum = (row-fw)/padw+1;
    hnum = (col-fh)/padh+1;
    for i = 1:hnum
        for j=1:wnum
            startx = 1+padw*(j-1);
            starty = 1+padh*(i-1);
            t = reshape( grad(:,(i-1)*wnum+j), fw, fh);
            gradient( startx:startx+fw-1, starty:starty+fh-1 ) = gradient( startx:startx+fw-1, starty:starty+fh-1 )+t;
        end
    end
end

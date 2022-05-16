import math
from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
        scores = np.dot(X[i], W)
        scores -= max(scores)
        sum_probs = np.sum(np.exp(scores))
        for j in range(num_classes):
            p = math.exp(scores[j]) / sum_probs
            dW[:, j] += (p - (j == y[i])) * X[i]
            if j == y[i]:
                loss += -math.log(p)

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W)
    # memo: numeric stability to avoid dividing large numbers due to the exponential
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # memo: softmax
    probs = np.exp(scores)
    sum_probs = np.sum(probs, axis=1, keepdims=True)
    softmax = np.divide(probs, sum_probs)
    
    # memo: data loss and regularization loss
    loss = -np.sum(np.log(softmax[np.arange(num_train), y])) / num_train \
                + reg * np.sum(W * W)
    
    # memo: mask for gradient calculation
    mask = np.copy(softmax)
    mask[np.arange(num_train), y] -= 1
    
    # memo: data gradient and regularization gradient
    dW = X.T.dot(mask) / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

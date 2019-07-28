from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    #record the margin for every class
    margin_list = np.zeros((1,10))
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                # correct class margin recorded as zero
                margin_list[0][j] = 0 
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            
            # Record the margin for j not equal to yi
            margin_list[0][j] = margin
            if margin > 0:
                loss += margin
                #Update the w of j class if the margin > 0 
                dW[:,j] += X[i]
        # calculate the total numer of margin which is greater than 0 
        dw_num = (margin_list > 0).sum()
        #Update the weight of yi column 
        dW[:,y[i]] -= dw_num * X[i]
      
        

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W) #elementwise
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


                

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    y_score = np.zeros(y.shape)
    CorrectClassScore_m = np.zeros((X.shape[0],W.shape[1]))
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #Compute the Score matrix, Score_m[i][j] refers to the score of ith example on the jth class 
    Score_m = np.dot(X,W)
    #y_score each row contains correct class score
    y_score = Score_m[np.arange(num_train),y]
#     print(Score_m)
#     print(y)
#     print(y_score)
    Margin_m = Score_m - y_score.reshape(-1,1)+ 1
    Margin_m[range(num_train), y] = 0
    Margin_m = (Margin_m > 0) * Margin_m
    #data loss
    loss += Margin_m.sum() / num_train
    #regularization loss
    loss += reg * np.sum(W * W)
   


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    counts = (Margin_m > 0).astype(int)
    counts[range(num_train), y] = - np.sum(counts, axis = 1)
    dW += np.dot(X.T, counts) / num_train + reg * W * 2
   

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

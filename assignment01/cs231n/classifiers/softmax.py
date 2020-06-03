import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg): #ppt39
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
    
  s = X.dot(W)#f(x,w)=wx
  num_train = X.shape[0]
  num_classe = W.shape[1]
 
  #softmax loss
  for i in range(num_train):
    f = s[i] - np.max(s[i]) # avoid numerical instability !!  
    softmax = np.exp(f)/np.sum(np.exp(f))
    loss += -np.log(softmax[y[i]]) #loss를 줄이기 위해 -사용
    
    # W Gradients
    for j in range(num_classe):
      dW[:,j] += X[i] * softmax[j]
    dW[:,y[i]] -= X[i] #정답 클래스에서는 image pixel value를 한번만 빼줌  
 
  loss /= num_train   #전체를 대표하는 평균 값
  dW /= num_train     
 
  # Regularization
  loss += reg * np.sum(W * W) #reg:regularization strength
  dW += reg * 2 * W  #!!!

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  s = X.dot(W)
   
  #단위 연산으로 빠르게(한번에 벡터 연산)
  # Softmax Loss
  sum_exp_s = np.exp(s).sum(axis=1, keepdims=True)
  softmax = np.exp(s)/sum_exp_s
  loss = np.sum(-np.log(softmax[np.arange(num_train), y]) ) #loss += -np.log(softmax[y[i]]) 
    #np.arange(num_train)  0~num_train-1

  # Weight Gradient
  softmax[np.arange(num_train),y] = softmax[np.arange(num_train),y]-1 
    #for j in range(num_classe)  ==> np.arange(num_train),y
  dW = X.T.dot(softmax) #dW[:,j] += X[i] * softmax[j]

  loss /= num_train
  dW /= num_train
 
  # Regularization
  loss += reg * np.sum(W * W) 
  dW += reg * 2 * W 


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


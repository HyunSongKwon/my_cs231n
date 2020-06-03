 
import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) 

  # loss and the gradient 구하기 
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
        #margin = scores[j] - scores[y[i]] + 1
      margin = scores[j] - correct_class_score + 1 
      if margin > 0:#0보다 작다면 ㅡ MAX값은 0이니까 아무것도 안하면 됨 
        loss += margin#전체 평균구하고 정규화할 때 쓰려고 더하는 듯
        dW[:, y[i]] = dW[:, y[i]] - X[i] #미분하면 x_j-x_yi니까
        dW[:,j] = dW[:,j] + X[i] 

  # loss: a sum over all training examples --> average(divide by num_train)
  loss /= num_train
  dW = dW / num_train 

  # regularizatio
  loss += reg * np.sum(W * W)
  dW = dW + reg * 2 * W 

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  scores = X.dot(W)#전체 label에 대한 score계산  shape:(500, 10)
  correct_class_scores = scores[ np.arange(num_train), y].reshape(num_train,1)
  #정답 label일때 scores가 얼마 나오는가   shape:(500, 1)

  
  margin = np.maximum(0, scores - correct_class_scores + 1)#정답을 맞췄다면 무조건 1이 되겠지
  margin[ np.arange(num_train), y] = 0 # loss에서 correct class 고려안하기 위해 0으로 만들기

  #average
  loss = margin.sum() / num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  
  ###########
 


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # Compute gradient
  margin[margin > 0] = 1 #정답 데이터보다 더 크게 나온 label값을 1로 만들기
  valid_margin_count = margin.sum(axis=1) #정답 데이터보다 더 크게 나온 label값 갯수만큼 더해짐

  # margin[np.arange(num_train),y--> correct class값이 0
  #margin = scores[j] - correct_class_score + 1  -->(delta = 1)  ,dW[:,j] += X[i,:],  dW[:,y[i]] -= X[i,:]
  # Subtract in correct class (-s_y)
  margin[np.arange(num_train),y ] -= valid_margin_count
  dW = (X.T).dot(margin) / num_train

  # Regularization gradient
  dW = dW + reg * 2 * W  #dw=dL/dw  (loss:reg * np.sum(W * W))
  ################
 

  return loss, dW

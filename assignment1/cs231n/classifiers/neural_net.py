from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    # print(W1.shape, b1.shape, W2.shape, b2.shape, X.shape)
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    O1 = X @ W1 + b1
    H1 = np.maximum(0, O1)
    O2 = H1 @ W2 + b2
    scores = O2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    exp_scores = np.exp(scores)
    exp_scores_sum = np.sum(exp_scores, axis=1)
    softmax_values = exp_scores / exp_scores_sum.reshape(-1, 1)
    loss = np.sum(-np.log(softmax_values[range(N), y]))
    loss /= N
    loss += reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(b1 ** 2) + np.sum(b2 ** 2))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dO2 = softmax_values
    dO2[range(N), y] = -1 + softmax_values[range(N), y]
    # print('dO2', dO2.shape)
    grads['b2'] = np.sum(dO2, axis=0)
    # print('b2', grads['b2'].shape)
    dM2 = dO2
    grads['W2'] = H1.T @ dM2
    # print('W2', grads['W2'].shape)
    dH1 = dM2 @ W2.T
    dO1 = dH1 * (O1 > 0)
    # print('dO1', dO1.shape)
    grads['b1'] = np.sum(dO1, axis=0)
    # print('b1', grads['b1'].shape)
    dM1 = dO1
    grads['W1'] = X.T @ dM1
    # print('W1', grads['W1'].shape)

    grads['W1'] /= N
    grads['b1'] /= N
    grads['W2'] /= N
    grads['b2'] /= N
    grads['W1'] += 2 * reg * W1
    grads['b1'] += 2 * reg * b1
    grads['W2'] += 2 * reg * W2
    grads['b2'] += 2 * reg * b2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(int(num_train / batch_size), 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      batch_mask = np.random.choice(np.arange(num_train), batch_size)
      X_batch = X[batch_mask, :]
      y_batch = y[batch_mask]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']
      self.params['b1'] -= learning_rate * grads['b1']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    O1 = X @ W1 + b1
    H1 = np.maximum(0, O1)
    O2 = H1 @ W2 + b2
    scores = O2
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred
  
def tune_parameters(X_train, y_train, X_val, y_val):
  best_net = None # store the best model into this
  best_val = -1

  input_size = X_train.shape[1]
  num_classes = 10

  hidden_size_set = [70]
  num_iters_set = [2500]
  batch_size_set = [256]
  learning_rate_set = [1e-3]
  learning_rate_decay_set = [0.95]
  reg_set = [0.3]

  # 50 3000 256 0.001 0.99 0.25  Validation accuracy:  0.494
  # 50 2000 256 0.001 0.95 0.25  Validation accuracy:  0.511

  # try #1
  # hidden_size_set = [25, 50, 100, 200]
  # num_iters_set = [1000]
  # batch_size_set = [32, 64, 200]
  # learning_rate_set = [1e-4, 5e-3, 1e-3]
  # learning_rate_decay_set = [0.9, 0.95, 0.99]
  # reg_set = [0.25, 2.5, 2.5e1]
  # 50 1000 200 0.001 0.9 0.25  Validation accuracy:  0.473
  # 50 1000 200 0.001 0.99 0.25  Validation accuracy:  0.479

  # try #2
  # hidden_size_set = [25, 50, 75]
  # num_iters_set = [1000, 2000]
  # batch_size_set = [256]
  # learning_rate_set = [1e-3, 2.5e-3, 5e-3]
  # learning_rate_decay_set = [0.99]
  # reg_set = [0.25]
  # 50 2000 256 0.001 0.99 0.25  Validation accuracy:  0.496

  # 50 3000 256 0.001 0.99 0.25  Validation accuracy:  0.494

  for hidden_size in hidden_size_set:
      for num_iters in num_iters_set:
          for batch_size in batch_size_set:
              for learning_rate in learning_rate_set:
                  for learning_rate_decay in learning_rate_decay_set:
                      for reg in reg_set:
                        print(hidden_size, num_iters, batch_size, learning_rate, learning_rate_decay, reg)
                        net = TwoLayerNet(input_size, hidden_size, num_classes)
                        
                        val_acc = train_during_tuning(
                          net, X_train, y_train, X_val, y_val, num_iters, batch_size, learning_rate, learning_rate_decay, reg
                        )
                        
                        print('\t\t\t\t\tValidation accuracy: ', val_acc)
                        if val_acc > best_val:
                            best_val = val_acc
                            best_net = net

  return best_net

def train_during_tuning(net, X_train, y_train, X_val, y_val, num_iters, batch_size, learning_rate, learning_rate_decay, reg):
  # Train the network
  stats = net.train(X_train, y_train, X_val, y_val,
              num_iters=num_iters, batch_size=batch_size,
              learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
              reg=reg, verbose=False)
  
  # Plot the loss function and train / validation accuracies
  print(stats['val_acc_history'])
  plt.subplot(2, 1, 1)
  plt.plot(stats['loss_history'])
  plt.title('Loss history')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')

  plt.subplot(2, 1, 2)
  plt.plot(stats['train_acc_history'], label='train')
  plt.plot(stats['val_acc_history'], label='val')
  plt.legend()
  plt.title('Classification accuracy history')
  plt.xlabel('Epoch')
  plt.ylabel('Clasification accuracy')
  plt.show()
  
  # Predict on the validation set
  val_acc = (net.predict(X_val) == y_val).mean()
  return val_acc
   
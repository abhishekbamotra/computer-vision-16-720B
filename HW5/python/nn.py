import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    high = np.sqrt(6 / (in_size + out_size))
    low = -1 * high
    
    W = np.random.uniform(low = low, high = high, size = (in_size, out_size))
    
    b = np.zeros(out_size)

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1 / (1 + np.exp(-x))

    ##########################
    ##### your code here #####
    ##########################

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################
    pre_act = np.matmul(X, W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    max_val_neg = - np.max(x, axis = 1)
    
    rep_mat = np.reshape(max_val_neg, (max_val_neg.size,1))
    exp_stable_x = np.exp(x - np.tile(rep_mat, x.shape[1]))
    
    sum_exp_x = np.sum(exp_stable_x, axis = 1)
    re_sum_exp_x = np.reshape(sum_exp_x, (sum_exp_x.size,1))
    
    tile_exp_x = np.tile(re_sum_exp_x, x.shape[1])
    res = np.divide(exp_stable_x, tile_exp_x)
    
    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################
    y_in_probs = np.multiply(y, np.log(probs))
    loss = -1*np.sum(y_in_probs)

    label_pred = np.argmax(probs, axis = 1)
    label_true = np.argmax(y, axis = 1)
    label_diff = label_pred - label_true
    
    count = 0
    for el in label_diff :
        if el == 0 :
            count += 1
            
    acc = count / probs.shape[0]

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################
    grad_X, grad_W, grad_b = np.zeros(X.shape), np.zeros(W.shape), np.zeros(b.shape)
    
    delta_post_act = activation_deriv(post_act)
    delta_pre_act = np.multiply(delta, delta_post_act)

    for i in range(X.shape[0]):
        x_reshape = X[i,:].reshape(X[i,:].size, 1)
        pre_act_re = delta_pre_act[i,:].reshape(1, delta_pre_act[i,:].size)
        
        grad_W += np.matmul(x_reshape, pre_act_re)
        grad_b += delta_pre_act[i,:]

        grad_X[i,:] = np.matmul(W, np.transpose(pre_act_re)).reshape([-1])

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################
    batches = list()
    idx = np.random.permutation(y.shape[0])
    
    for i in range(0, y.shape[0], 5):
        
        batches.append((x[idx[i:i+5],:], y[idx[i:i+5],:]))
        
    return batches

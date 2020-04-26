import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(train_x.shape[1],hidden_size,params, 'layer1')
initialize_weights(hidden_size,hidden_size,params,'layer2')
initialize_weights(hidden_size,hidden_size,params,'layer3')
initialize_weights(hidden_size,train_x.shape[1],params,'decode1')

train_loss, valid_loss = list(), list()
n_layers = len(params) // 2 - 1

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        #Forward pass
        frwd1 = forward(xb, params, 'layer1', activation = relu)
        frwd2 = forward(frwd1, params, 'layer2', activation = relu)
        frwd3 = forward(frwd2, params, 'layer3', activation = relu)
        output = forward(frwd3, params, 'decode1', activation=sigmoid)

        #Loss calculation
        total_loss += np.sum((xb - output)**2)
        
        #Back pass
        back1 = backwards(-2 * (xb - output), params, 'decode1', sigmoid_deriv)
        back2 = backwards(back1, params, 'layer3', relu_deriv)
        back3 = backwards(back2, params, 'layer2', relu_deriv)
        backwards(back3, params, 'layer1', relu_deriv)
        
        for idx in range(n_layers):
            str_layer = 'layer' + str(idx + 1)
            
            params['m_W' + str_layer] = 0.9 * params['m_W' + str_layer] - learning_rate * params['grad_W' + str_layer]
            params['m_b' + str_layer] = 0.9 * params['m_b' + str_layer] - learning_rate * params['grad_b' + str_layer]
            params['W' + str_layer] += params['m_W' + str_layer]            

        params['m_Wdecode1'] = 0.9 * params['m_Wdecode1'] - learning_rate * params['grad_Wdecode1']
        params['m_bdecode1'] = 0.9 * params['m_bdecode1'] - learning_rate * params['grad_bdecode1']
        params['Wdecode1'] += params['m_Wdecode1']
        
    total_loss = total_loss / train_x.shape[0]
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
        
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
        
    frwd1 = forward(valid_x, params, 'layer1', activation = relu)
    frwd2 = forward(frwd1, params, 'layer2', activation = relu)
    frwd3 = forward(frwd2, params, 'layer3', activation = relu)
    out_valid = forward(frwd3, params, 'decode1', activation=sigmoid)
    
    v_loss = np.sum(np.square(valid_x - out_valid)) / valid_x.shape[0]
    
    train_loss.append(total_loss)
    valid_loss.append(v_loss)


x_val = range(max_iters)

plt.figure('loss')
plt.plot(x_val, train_loss)
plt.plot(x_val, valid_loss)
plt.legend(['train', 'validation'])
plt.xlabel('iterations')
plt.ylabel('loss')
plt.title('Q5')
plt.show()
        
# Q5.3.1
# import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################

for idx in [100, 120, 500, 534, 1200, 1222, 2110, 2140, -10, -1]:
    org_img = valid_x[idx].reshape(32,32).T
    reconst_img = out_valid[idx].reshape(32,32).T
    plt.imshow(org_img)
    plt.show()
    plt.imshow(reconst_img)
    plt.show()


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
valPSNR = 0
for i in range(valid_x.shape[0]) :
    valPSNR += psnr(valid_x[i], out_valid[i])
    
print ('Average PSNR is: ', valPSNR / valid_x.shape[0])

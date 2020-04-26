import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.002
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,36,params,'output')

train_loss, train_acc = list(), list()
valid_loss, valid_acc = list(), list()


# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    count = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        frwd1 = forward(xb, params, 'layer1')
        pred1 = forward(frwd1, params, 'output', softmax)
        loss_out, acc_out = compute_loss_and_acc(yb, pred1)
        
        total_loss +=  loss_out
        total_acc += acc_out
        
        del_val = backwards(pred1 - yb,params, 'output', linear_deriv)
        backwards(del_val, params, 'layer1', sigmoid_deriv)

        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['boutput'] -= learning_rate * params['grad_boutput']
        count += 1

    total_acc = total_acc / batch_num
    total_loss = total_loss / train_x.shape[0]
    train_loss.append(total_loss)
    train_acc.append(total_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
    
    frwd_v = forward(valid_x, params, 'layer1')
    pred_v = forward(frwd_v, params, 'output', softmax)
    loss_valid, acc_valid = compute_loss_and_acc(valid_y, pred_v)
    
    valid_loss.append(loss_valid / valid_x.shape[0])
    valid_acc.append(acc_valid)

x_values = range(max_iters)

plt.figure('accuracy')
plt.plot(x_values, train_acc)
plt.plot(x_values, valid_acc)
plt.legend(['train', 'validation'])
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title('learning rate ' + str(learning_rate))
plt.show()

plt.figure('loss')
plt.plot(x_values, train_loss)
plt.plot(x_values, valid_loss)
plt.legend(['train', 'validation'])
plt.xlabel('iterations')
plt.ylabel('loss')
plt.title('learning rate ' + str(learning_rate))
plt.show()


# run on validation set and report accuracy! should be above 75%
print('Validation accuracy: ', valid_acc[-1])

frwd_t = forward(test_x, params, 'layer1')
pred_t = forward(frwd_t, params, 'output', softmax)
loss_test, acc_test = compute_loss_and_acc(test_y, pred_t)
# valid_acc = None
##########################
##### your code here #####
##########################
print('Test accuracy: ',acc_test)


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    after_init = saved_params['Wlayer1']
    
init_hidden_weights = params['Wlayer1']
# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################

## Init Weights
fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                  nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                  axes_pad=0.1,  # pad between axes in inch.
                  )

for i in range(init_hidden_weights.shape[1]):
    layerImage = np.reshape(init_hidden_weights[:,i], (32, 32))
    grid[i].imshow(layerImage)  
plt.title('initialize weights')
plt.show()

## After init weights
fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                  nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                  axes_pad=0.1,  # pad between axes in inch.
                  )

for i in range(after_init.shape[1]):
    layerImage = np.reshape(after_init[:,i], (32, 32))
    grid[i].imshow(layerImage)
plt.title('immediately after initialize weights')
plt.show()


# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################

##### Train Data #####
frwd1 = forward(train_x, params, 'layer1')
pred1 = forward(frwd1, params, 'output', softmax)
true_y = np.argmax(train_y, axis = 1)
pred_y = np.argmax(pred1, axis = 1)

for i in range(true_y.size) :
    confusion_matrix[true_y[i]][pred_y[i]] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.title('Train Data')
plt.show()

##### Validation Data #####
frwd1 = forward(valid_x, params, 'layer1')
pred1 = forward(frwd1, params, 'output', softmax)
true_y = np.argmax(valid_y, axis = 1)
pred_y = np.argmax(pred1, axis = 1)

for i in range(true_y.size) :
    confusion_matrix[true_y[i]][pred_y[i]] += 1

plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.title('Validation Data')
plt.show()

##### Test Data #####
frwd1 = forward(test_x, params, 'layer1')
pred1 = forward(frwd1, params, 'output', softmax)
true_y = np.argmax(test_y, axis = 1)
pred_y = np.argmax(pred1, axis = 1)

for i in range(true_y.size) :
    confusion_matrix[true_y[i]][pred_y[i]] += 1

plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.title('Test Data')
plt.show()
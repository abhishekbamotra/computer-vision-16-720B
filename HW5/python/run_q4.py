import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    data_arr = np.array([])
    cluster_cent = dict([])
    threshold = 150
    
    for idx in range(bboxes.shape[0]):
        box = bboxes[idx,:]
        
        box_cord = (box[0] + box[2]) / 2

        count = 0
        for key, values in cluster_cent.items():
            
            if abs(box_cord - key) > threshold:
                count += 1
            else:
                cluster_cent[key] = np.vstack((cluster_cent[key], np.array(box)))

        if count == len(cluster_cent) :
            cluster_cent[box_cord] = np.array(box)
            
            
            
    for key, values in cluster_cent.items():
        ent_arr = np.array([])
        idx_order = np.argsort(values[:,1])
        
        for el in idx_order:
            
            if ent_arr.size == 0:
                ent_arr = values[el]    
            else:
                ent_arr = np.vstack((ent_arr, values[el]))
                
        cluster_cent[key] = ent_arr


    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    idx_leave = np.array([])
    el_count = 0
    
    for key, value in cluster_cent.items() :
        #bboxVal = value
        for el in value :
            img_reg = bw[el[0]:el[2], el[1]:el[3]]

            if img_reg.shape[0] > img_reg.shape[1]:
                diff = img_reg.shape[0] - img_reg.shape[1]
                img_reg = np.pad(img_reg, ((10,10),(diff // 2, diff // 2)), 'constant', constant_values=1)
                
            elif img_reg.shape[0] < img_reg.shape[1]:
                diff = img_reg.shape[1] - img_reg.shape[1]
                img_reg = np.pad(img_reg, ((diff // 2, diff // 2), (10,10)), 'constant', constant_values=1)
                
            img_reg = skimage.transform.resize(img_reg, (32, 32))
            img_reg = skimage.morphology.erosion(img_reg, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])).T
            img_vec = img_reg.flatten().reshape(1, img_reg.size)
            
            if data_arr.size == 0 :
                data_arr = img_vec
            else:
                data_arr = np.vstack((data_arr, img_vec))
            el_count += 1
        idx_leave = np.append(idx_leave, el_count)
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    frwd1 = forward(data_arr, params, 'layer1')
    pred1 = forward(frwd1, params, 'output', softmax)
    pred_y = np.argmax(pred1, axis = 1)
    
    out = ""
    for idx in range(pred_y.size) :
        if idx in idx_leave :
            print(out)
            out = ""
        out += letters[pred_y[idx]]
    print(out)
    print('================')
    
import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import visual_words
import sklearn


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    hist = np.histogram(wordmap.reshape((wordmap.shape[0]*wordmap.shape[1], 1)), bins = range(0, K+1))
    hist_result = hist[0] / np.linalg.norm(hist[0], ord=1)
    hist_result = np.reshape(hist_result, (1,K))
    return hist_result

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    L -= 1
    
    # ----- TODO -----
    wm_shape = wordmap.shape
    keep = wm_shape[0] if wm_shape[0] < wm_shape[1] else wm_shape[1]
    drop = keep % (2**L)
    
    hist_list = np.array([]).reshape((1,0))
    for l in range(L, -1, -1):
        finest_split = np.array_split(wordmap[:keep-drop, :keep-drop], 2**(2*l))
        for elem in finest_split:
            temp = get_feature_from_wordmap(opts, elem)
            
            if (l == 0 or l == 1):
                weight = 1/8
            else:
                weight = 3/4
                
            histo = np.asarray(temp*weight)

            hist_list = np.hstack((hist_list, histo))
    return hist_list
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    features = get_feature_from_wordmap_SPM(opts, wordmap)
    
    return features

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    size = int((opts.K * (4**SPM_layer_num - 1) / 3))
    print(size)
    pool = multiprocessing.Pool(n_worker)
    features = np.array([]).reshape((0,size))
    print(len(train_files))
    # for img_path in train_files:
    opts_list = [opts for i in range(len(train_files))]
    dict_list = [dictionary for i in range(len(train_files))]
    paths_list = [join(data_dir, j) for j in train_files]
    
    args = list(zip(opts_list, paths_list, dict_list))
    all_list = pool.starmap(get_image_feature,args)
    
    features = np.vstack(all_list)    
    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )
    
    return True

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    mini = np.minimum(word_hist, histograms)
    sim = np.sum(mini, axis = 1)
    return sim
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    features = trained_system['features']
    train_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    n_test = len(test_files)
	# ----- TODO -----
    
    pool = multiprocessing.Pool(n_worker)
    # for img_path in train_files:
    opts_list = [opts for i in range(n_test)]
    dict_list = [dictionary for i in range(n_test)]
    test_paths_list = [join(data_dir, j) for j in test_files]
    
    args = list(zip(opts_list, test_paths_list, dict_list))
    all_list = pool.starmap(get_image_feature,args)
    test_features = np.vstack(all_list)  
    
    test_pred = list()
    for t in test_features:
        similarity = distance_to_set(t, features)
        label_pred = train_labels[np.argmax(similarity)]
        test_pred.append(label_pred)

    confusion = np.zeros((8,8))
    
    count = 0
    for i in range(len(test_pred)):
        x = test_labels[i]
        y = test_pred[i]
        confusion[x, y] += 1
        if test_pred[i] == test_labels[i]:
            count += 1
    accuracy = count/len(test_pred)
    
    return (confusion, accuracy)


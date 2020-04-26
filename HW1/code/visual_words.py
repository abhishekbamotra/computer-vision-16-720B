import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import skimage.color
from sklearn.cluster import KMeans
import scipy


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    
    if(img.dtype != np.float32 and np.max(img) <= 1 and np.min(img) >= 0):
        img = img/255.0
        
    if(len(img.shape) != 3):
        img = np.dstack((img, img, img))
    
    labImg = skimage.color.rgb2lab(img)
    filter_responses = np.empty(labImg.shape)
    for sig in filter_scales:
        
        # Gaussian Filter
        gauss1 = ndi.gaussian_filter(labImg[:,:,0], sig, mode='reflect')
        gauss2 = ndi.gaussian_filter(labImg[:,:,1], sig, mode='reflect')
        gauss3 = ndi.gaussian_filter(labImg[:,:,2], sig, mode='reflect')
        gauss = np.dstack((gauss1, gauss2, gauss3))
        filter_responses = np.append(filter_responses, np.atleast_3d(gauss), axis=2)
        
        # Laplacian of Gaussian Filter
        lap1 = ndi.gaussian_laplace(labImg[:,:,0], sig, mode='reflect')
        lap2 = ndi.gaussian_laplace(labImg[:,:,1], sig, mode='reflect')
        lap3 = ndi.gaussian_laplace(labImg[:,:,2], sig, mode='reflect')
        lap = np.dstack((lap1, lap2, lap3))
        filter_responses = np.append(filter_responses, np.atleast_3d(lap), axis=2)
        
        # Derivative of gaussian filter in x direction
        gaussDx1 = ndi.gaussian_filter(labImg[:,:,0], sig, order= (1,0), mode='reflect')
        gaussDx2 = ndi.gaussian_filter(labImg[:,:,1], sig, order= (1,0), mode='reflect')
        gaussDx3 = ndi.gaussian_filter(labImg[:,:,2], sig, order= (1,0), mode='reflect')
        gaussDx = np.dstack((gaussDx1, gaussDx2, gaussDx3))
        filter_responses = np.append(filter_responses, np.atleast_3d(gaussDx), axis=2)
        
        # Derivative of gaussian filter in y direction 
        gaussDy1 = ndi.gaussian_filter(labImg[:,:,0], sig, order= (0,1), mode='reflect')
        gaussDy2 = ndi.gaussian_filter(labImg[:,:,1], sig, order= (0,1), mode='reflect')
        gaussDy3 = ndi.gaussian_filter(labImg[:,:,2], sig, order= (0,1), mode='reflect')
        gaussDy = np.dstack((gaussDy1, gaussDy2, gaussDy3))
        filter_responses = np.append(filter_responses, np.atleast_3d(gaussDy), axis=2)
        
    # ----- TODO -----
    filter_responses = filter_responses[:, :, 3:]
    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    opts, alpha, path = args
    img = np.array(Image.open(path)).astype(np.float32)/255
    respo = extract_filter_responses(opts = opts, img = img)
    reshaped_respo = respo.reshape((img.shape[0]*img.shape[1], 1, respo.shape[2]))
    filter_responses = np.random.permutation(reshaped_respo)[:alpha, :, :]
    filter_responses = filter_responses.reshape((alpha, respo.shape[2]))

    return filter_responses

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    print(K)
    alpha = opts.alpha
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    
    paths_list = [join(data_dir, i) for i in train_files]
    
    pool = multiprocessing.Pool(n_worker)

    args = [(opts, alpha, j) for j in paths_list]
    feat = pool.map(compute_dictionary_one_image, args)
    
    temp = feat[0]
    for i in range(1, len(feat)):
        temp = np.concatenate((temp, feat[i]), axis=0)
    # save output features
    np.save('../filtered_responses.npy', temp)
    temp = np.load('../filtered_responses.npy')
	# perform k-means clustering
    k_means = KMeans(n_clusters=K, n_jobs=n_worker).fit(temp)
    dictionary = k_means.cluster_centers_
    print (dictionary.shape)
    
    ## example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    img_shape = img.shape
    f_r = extract_filter_responses(opts, img)
    f_responses = f_r.reshape((f_r.shape[0]*f_r.shape[1], 1, f_r.shape[2]))
    wordmap = np.zeros(img_shape[0]*img_shape[1])
    for j in range(len(f_responses)):
      
        dist = scipy.spatial.distance.cdist(f_responses[j], dictionary, metric='euclidean')
        wordmap[j] = np.argmin(dist)
 
    wordmap = wordmap.reshape((img_shape[0], img_shape[1]))

    return wordmap


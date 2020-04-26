import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

import matplotlib.pyplot as plt
import matplotlib.patches as patches
# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = np.array([]) #[]
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    im = skimage.color.rgb2gray(image)
    gauss_im = skimage.filters.gaussian(im, sigma=1.0)
    thres = skimage.filters.threshold_otsu(gauss_im)
    thres_pass_im = gauss_im <= thres
    cut = skimage.morphology.closing(thres_pass_im, skimage.morphology.square(10))

    label_im = skimage.measure.label(cut, connectivity = 1)
    reg_to = skimage.measure.regionprops(label_im)
    
    count = 0
    for reg in reg_to:
    	count += 1
        
    	if count >= 1:
    		if reg.area >= 400:
                
    			if bboxes.size == 0:
    				bboxes = np.array(np.array(reg.bbox))
    			else:
    				bboxes = np.vstack((bboxes, np.array(reg.bbox)))

    bw =  (~(cut)).astype(np.float)

    return bboxes, bw






#### Run findLetter ####

if __name__ == "__main__":
    paths = ['01_list.jpg', '02_letters.jpg', '03_haiku.jpg', '04_deep.jpg']
    for el in range(len(paths)):

        im = skimage.io.imread('../images/' + paths[el])
        bb, bw = findLetters(im)
        print("Image: " + paths[el])
        print(bb)
        
        fig,ax = plt.subplots(1)
        ax.imshow(im)
    
        for i in range(bb.shape[0]) :
            rectShow = patches.Rectangle((bb[i,1],bb[i,0]),bb[i,3] - bb[i,1],bb[i,2] - bb[i,0], linewidth=2,edgecolor='red', facecolor='none')
            ax.add_patch(rectShow)
        plt.show()
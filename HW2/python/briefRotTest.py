import numpy as np
import cv2
from matchPics import matchPics
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from helper import plotMatches

#Q2.1.6
#Read the image and convert to grayscale, if necessary
from opts import get_opts

opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')

matches_hist = list()

for i in range(36):
    #Rotate Image

    rotated_img = rotate(cv_cover, 10*(i))
	#Compute features, descriptors and Match features
    
    matches, locs1, locs2 = matchPics(cv_cover, rotated_img, opts)
	#Update histogram
    matches_hist.append(len(matches))
    
    if(i%11 == 0):
        plotMatches(cv_cover, rotated_img, matches, locs1, locs2)

print(matches_hist)
#Display histogram
m_hist = np.asarray(matches_hist)
ang = np.asarray([i for i in range(0, 360, 10)]).T

y = np.arange(len(ang))
plt.bar(y, m_hist.T, align='center')
plt.xticks(y, ang)
plt.ylabel('matches')


import cv2
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
from helper import plotMatches

#Write script for Q2.2.4
opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

x1 = locs1[matches[:,0], 0:2]
x2 = locs2[matches[:,1], 0:2]

H, inliers = computeH_ransac(x1, x2, opts)

hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1],cv_cover.shape[0]))
result = compositeH(H, hp_cover, cv_desk)

cv2.imwrite('hp_result.jpeg',result)
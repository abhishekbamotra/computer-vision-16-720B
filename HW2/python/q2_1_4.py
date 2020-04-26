import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts


opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')


matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
#display matched features
plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

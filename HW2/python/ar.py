import cv2

#Import necessary functions
from loadVid import loadVid
from matchPics import matchPics
from opts import get_opts
from planarH import computeH_ransac
from planarH import compositeH


def func_result(cv_cover, frame, ar_f, opts):
    matches, locs1, locs2 = matchPics(cv_cover, frame, opts)
    x1 = locs1[matches[:,0], 0:2]
    x2 = locs2[matches[:,1], 0:2]
    
    H2to1, inliers = computeH_ransac(x1, x2, opts)
    ar_f = ar_f[45:310,:,:]
    cover_width = cv_cover.shape[1]
    width = int(ar_f.shape[1]/ar_f.shape[0]) * cv_cover.shape[0]

    resized_ar = cv2.resize(ar_f, (width,cv_cover.shape[0]), interpolation = cv2.INTER_AREA)
    h, w, d = resized_ar.shape
    cropped_ar= resized_ar[:,int(w/2)-int(cover_width/2):int(w/2)+int(cover_width/2),:]
    
    result = compositeH(H2to1, cropped_ar, frame)
    
    return result

opts = get_opts()

book = loadVid('../data/book.mov')
src = loadVid('../data/ar_source.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')


a, b, _ = book[1].shape
out = cv2.VideoWriter('ar.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, (b, a))

for i in range(src.shape[0]):
    frame = book[i]
    ar_f = src[i]
    print(i)
    result = func_result(cv_cover, frame, ar_f, opts)
    out.write(result)

cv2.destroyAllWindows()
out.release()

import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    # Put your implementation here
    p = p0
    x1, y1, x2, y2 = rect

    template_width = int(x2 - x1)
    template_height = int(y2 - y1)
    x, y = np.mgrid[x1:x2+1:template_width*1j, y1:y2+1:template_height*1j]
    
    h1, w1 = It.shape
    h2, w2 = It1.shape

    It_spl = RectBivariateSpline(np.linspace(0, h1, num=h1, endpoint=False), np.linspace(0, w1, num=w1, endpoint=False), It)
    It1_spl = RectBivariateSpline(np.linspace(0, h2, num=h2, endpoint=False), np.linspace(0, w2, num=w2, endpoint=False), It1)

    thres = np.inf
    iters = 0
    
    while (thres > threshold) and (iters < num_iters):
        iters += 1 
        
        y_new = y+p[1]
        x_new = x+p[0]
        
        delX_spl_out = It1_spl.ev(y_new, x_new, dy=1).flatten()
        delY_spl_out = It1_spl.ev(y_new, x_new, dx=1).flatten()
        
        It_spl_out = It_spl.ev(y, x).flatten()
        It1_spl_out = It1_spl.ev(y_new, x_new).flatten()

        A = np.hstack((np.reshape(delX_spl_out, (-1, 1)), np.reshape(delY_spl_out, (-1, 1))))
        b = np.reshape(It_spl_out-It1_spl_out, (template_width*template_height, 1))
        
        dp = np.linalg.pinv(A) @ b
        p = (p + dp.T).ravel()
        
        thres = np.linalg.norm(dp)
    return p

    
    
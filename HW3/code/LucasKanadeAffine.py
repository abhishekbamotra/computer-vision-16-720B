import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    It_height, It_width = It.shape
    It1_height, It1_width = It1.shape
    template_width = It_width
    template_height = It_height
    
    x, y = np.mgrid[0:It_width, 0:It_height]
    x = x.reshape(1, template_height*template_width)
    y = y.reshape(1, template_height*template_width)
    
    It_spl = RectBivariateSpline(np.linspace(0, It_height, num=It_height, endpoint=False), np.linspace(0, It_width, num=It_width, endpoint=False), It)
    It1_spl = RectBivariateSpline(np.linspace(0, It1_height, num=It1_height, endpoint=False), np.linspace(0, It1_width, num=It1_width, endpoint=False), It1)

    dp = np.inf
    iters = 0
    
    homo = np.vstack((x, y, np.ones((1, template_height*template_width))))
    p = np.zeros(6)

    while dp > threshold and iters < num_iters:
        iters += 1
        
        M = np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]]])
        warp_x, warp_y = M @ homo

        outX = (np.where(warp_x >= It_width) or np.where(warp_x < 0))
        outY = (np.where(warp_y >= It_height) or np.where(warp_y < 0))

        chk1 = np.shape(outX)[1]
        chk2 = np.shape(outY)[1]
        
        if chk1 == 0 and chk2 == 0:
            outter = list()
        elif chk1 == 0 and chk2 != 0:
            outter = outY
        elif chk1 != 0 and chk2 == 0:
            outter = outX
        else:
            outter = np.unique(np.concatenate((outX, outY), 0))

        x_new, y_new, warp_x, warp_y = np.delete(x, outter), np.delete(y, outter), np.delete(warp_x, outter), np.delete(warp_y, outter)
        
        gradX = It1_spl.ev(warp_y, warp_x, dy=1).flatten()
        gradY = It1_spl.ev(warp_y, warp_x, dx=1).flatten()
        It_spl_out = It_spl.ev(y_new, x_new).flatten()
        It1_spl_out = It1_spl.ev(warp_y, warp_x).flatten()
        
        x_new = x_new.reshape(-1, 1)
        y_new = y_new.reshape(-1, 1)
        
        gradX = gradX.reshape(-1, 1)
        gradY = gradY.reshape(-1, 1)
        
        warp_x = warp_x.reshape(-1, 1)
        warp_y = warp_y.reshape(-1, 1)
        
        A = np.hstack((x_new*gradX, y_new*gradX, gradX, x_new*gradY, y_new*gradY, gradY))
        b = np.reshape(It_spl_out-It1_spl_out, (-1, 1))

        del_P = np.linalg.pinv(A) @ b
        p = (p + del_P.T).flatten()
        
        dp = np.linalg.norm(del_P)
        
    M = np.array([[1+p[0], p[1], p[2]],
                  [p[3], 1+p[4], p[5]]])
    return M

import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage as nd

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gradY, gradX = np.gradient(It)
    
    iters = 0
    dp = np.inf
    scale = np.array([[0.0,0.0,1.0]])
    
    while iters < num_iters and dp < threshold:
        iters += 1
        
        temp = np.copy(M)
        temp[0:2, 0:2] = np.fliplr(temp[0:2, 0:2])
        temp = np.vstack((np.flipud(temp), scale))
        
        wrap_It1 = nd.affine_transform(It1, temp, cval = -1)
        c_pts = np.where(wrap_It1 != -1)
        B_vec = wrap_It1[c_pts] - It[c_pts]
        
        A_mat = np.zeros([B_vec.size, 6])
        A_mat[:,0:3] = gradX[c_pts] * c_pts[1], gradX[c_pts] * c_pts[0], gradX[c_pts]
        A_mat[:,3:6] = gradY[c_pts] * c_pts[1], gradY[c_pts] * c_pts[0], gradY[c_pts]
        
        del_P = np.linalg.lstsq(A_mat, B_vec, rcond = -1)[0]
        dp = np.linalg.norm(del_P)
        del_P = np.reshape(del_P, (2,3))
        
        del_P += np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        del_P = np.vstack((del_P, scale))

        M = np.vstack((M, scale))
        M = (M @ np.linalg.inv(del_P))[0:2, :]
        
    return M

from scipy import ndimage as nd
import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    # put your implementation here
    mask = np.zeros(image1.shape, dtype=bool)
    
    # M_mat = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M_mat = InverseCompositionAffine(image1, image2, threshold, num_iters)

    warped = nd.affine_transform(image1, -M_mat, output_shape=None, offset=0.0)
    diff = abs(warped - image2)
    
    mask[diff < tolerance] = 0
    mask[diff > tolerance] = 1

    mask = nd.morphology.binary_erosion(mask)
    mask = nd.morphology.binary_dilation(mask, iterations=1)
    
    return mask

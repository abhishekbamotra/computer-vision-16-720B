import numpy as np

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    
    image_output = np.zeros(output_shape)
 
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):

                transf_location = np.array([i, j, 1]).reshape(3,1)
                trans_mat = np.linalg.inv(A) @ transf_location
                
                new_i = int(round(trans_mat[0][0]))
                new_j = int(round(trans_mat[1][0]))

                if new_i < 0 or new_j < 0 or new_i > 199 or new_j > 149:
                    image_output[i, j] = 0
                else:
                    image_output[i, j] = im[new_i, new_j]
    
    return image_output
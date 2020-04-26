'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import submission
import helper
import matplotlib.pyplot as plt

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
dataK = np.load('../data/intrinsics.npz')

N = data['pts1'].shape[0]
M = 640

F = submission.eightpoint(data['pts1'], data['pts2'], M)
E = submission.essentialMatrix(F, dataK['K1'], dataK['K2'])

Ms = helper.camera2(E)

M_mat = np.eye(3)
M_mat = np.hstack((M_mat, np.zeros((3,1))))

C1 = np.dot(dataK['K1'], M_mat)

err = np.inf

for idx in range(Ms.shape[2]):
    
    C2 = np.dot(dataK['K2'], Ms[:,:,idx])
    
    P_calc, err_calc = submission.triangulate(C1, data['pts1'], C2, data['pts2'])
    
    if err_calc < err:
        err = err_calc
        M2_mat = Ms[:,:,idx]
        C2_mat = C2
        P = P_calc

# print(M2_mat, C2_mat)
np.savez('q3_3.npz', M2=M2_mat, C2=C2_mat, P=P)

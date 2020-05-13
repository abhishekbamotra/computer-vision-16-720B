# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    U,S,Vt = np.linalg.svd(I, full_matrices=False)
    S[3:] = 0
    S3 = np.diag(S[:3])
    VT3 = Vt[:3,:]
    B = np.dot(np.sqrt(S3),VT3)
    L = (U[:,:3] @ np.sqrt(S3)).T
    return B, L


if __name__ == "__main__":

    ########## Ques 2.b ##########
    # # Put your main code here
    # I, L_true, s = loadData()
    # B, L = estimatePseudonormalsUncalibrated(I)
    # print(L_true)
    # print(L)
    # a, n = estimateAlbedosNormals(B)
    # alb, nor = displayAlbedosNormals(a, n, s)
    
    # ########## Ques 2.d ##########
    # # Put your main code here
    # I, L_true, s = loadData()
    # B, L = estimatePseudonormalsUncalibrated(I)

    # a, n = estimateAlbedosNormals(B)
    # surface = estimateShape(n, s)
    
    # min_v, max_v = np.min(surface), np.max(surface)
    # surface = (surface - min_v) / (max_v - min_v)
    
    # surface = (surface * 255.).astype('uint8')
    # plotSurface(surface)
    
    ########## Ques 2.e ##########
    # Put your main code here
    I, L_true, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)

    integ_normal = enforceIntegrability(B, s)
    a, n = estimateAlbedosNormals(integ_normal)
    surface = estimateShape(n, s)
    
    min_v, max_v = np.min(surface), np.max(surface)
    surface = (surface - min_v) / (max_v - min_v)
    
    surface = (surface * 255.).astype('uint8')
    plotSurface(surface)
    
    
    # ########## Ques 2.f ##########
    # I, L_true, s = loadData()
    # B, L = estimatePseudonormalsUncalibrated(I)
 
    # # mu=8 #faltten as increased
    # # v=0.001
    # # lambda=0.01 #gives more curve as increased
    # G=np.array([[1,0,0],[0,1,0],[8,0.001,0.01]])
    
    # print(G)
    # G_invT =np.linalg.inv(G).T
    # integ_normal = enforceIntegrability(B, s)
    # a, n = estimateAlbedosNormals(np.dot(G_invT,integ_normal))
    
    # surface = estimateShape(n, s)
    
    # min_v, max_v = np.min(surface), np.max(surface)
    # surface = (surface - min_v) / (max_v - min_v)
    
    # surface = (surface * 255.).astype('uint8')
    # plotSurface(surface)
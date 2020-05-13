# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import skimage
from matplotlib import cm
from mpl_toolkits import mplot3d

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    albedo = 0.5
    
    X = np.linspace(0, res[0]-1, int(res[0]))
    Y = np.linspace(0, res[1]-1, int(res[1]))
    [x,y] = np.meshgrid(X, Y)
    
    rad = int(rad // pxSize)
    resd = rad**2 - ((res[0]//2 + int(center[0]//pxSize) - x)**2 + (res[1]//2 - int(center[1]//pxSize) - y)**2)
    mask = (resd >= 0)
    resd = resd * mask
    
    index = np.where(resd == 0)
    resd[index[0],index[1]] = 1
    
    p = (x-res[0]//2) / np.sqrt(resd)
    q = (y-res[1]//2) / np.sqrt(resd)
    
    R = (albedo * (light[0] * p - light[1] * q + light[2])) / np.sqrt(1 + p**2 + q**2)
    R = R * mask

    index = np.where(R < 0)
    R[index[0], index[1]] = 0
    
    image = R / np.max(R) * 255
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    img = skimage.io.imread(path+'input_1.tif')
    I = np.zeros((7,img.shape[0]*img.shape[1]))
    for i in range(1,8):
        
        img = skimage.io.imread(path+'input_'+str(i)+'.tif')
        
        if isinstance(img[0,0,0], np.uint16):
            xyz = skimage.color.rgb2xyz(img)
            y = xyz[:,:,1].astype(np.float32)
            I[i-1,:] = y.flatten()
    
    sources = np.load(path+'sources.npy')
    L = sources.T
    s = img.shape[:2]
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    
    B = np.linalg.lstsq(L.T, I, rcond=None)[0]
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    albedos = list()
    
    for i in range(B.shape[1]):
        mag = np.linalg.norm(B[:,i])
        albedos.append(mag)
        B[:,i] /= mag
    
    normals = B
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = np.reshape(albedos, s)
    plt.imshow(albedoIm, cmap='gray')
    plt.show()

    normals += abs(np.min(normals))
    normals /= np.max(normals)
    l1 = normals[0,:].reshape(s)
    l2 = normals[1,:].reshape(s)
    l3 = normals[2,:].reshape(s)
    normalIm = np.dstack((np.dstack((l1, l2)), l3))
    plt.imshow(normalIm, cmap='rainbow')
    plt.show()
    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    zx = np.reshape(normals[0,:] / (-normals[2,:]), s)
    zy = np.reshape(normals[1,:] / (-normals[2,:]), s)
    surface = integrateFrankot(zx, zy)
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    h, w = surface.shape
    y, x = range(h), range(w)
    fig = plt.figure()
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, surface, edgecolor='none', cmap=cm.coolwarm)
    ax.set_title('Surface plot')
    plt.show()


if __name__ == '__main__':

    # Put your main code here
    # pass
    #### Ques 1b ####
    # light = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
    # light = np.array([1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)])
    # light = np.array([-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)])
    # light = np.array([0, 0, 1])
    # center = np.array([0,0,0])
    # radius = 0.75
    # frame = np.array([3840, 2160])
    
    # img = renderNDotLSphere(center, radius, light, 0.0007, frame)

    # plt.imshow(img, cmap='gray')
    # plt.show()
    
    #### Ques 1d ####
    I, L, s = loadData()
    
    u_svd,s_svd,vt_svd = np.linalg.svd(I, full_matrices=False)
    print(s_svd)
    
    B = estimatePseudonormalsCalibrated(I,L)
    a, n = estimateAlbedosNormals(B)

    # alb, nor = displayAlbedosNormals(a, n, s)
    
    
    surface = estimateShape(n, s)
    
    min_v, max_v = np.min(surface), np.max(surface)
    surface = (surface - min_v) / (max_v - min_v)
    
    surface = (surface * 255.).astype('uint8')

    plotSurface(surface)
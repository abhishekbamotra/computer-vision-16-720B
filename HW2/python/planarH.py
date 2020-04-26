import numpy as np
import cv2


def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
    A_mat = np.zeros((2*x1.shape[0], 9))

    for i in range(0, x1.shape[0]):
        a = x1[i, 0]
        b = x1[i, 1]
        p = x2[i, 0]
        q = x2[i, 1]
        
        A_mat[2*i, :] = np.array([-p, -q, -1, 0, 0, 0, a*p, a*q, a])
        A_mat[(2*i+1), :] = np.array([0, 0, 0, -p, -q, -1, b*p, b*q, b])
        
    U, S, V = np.linalg.svd(A_mat)
    h = V[-1, :]
  
    H2to1 = h.reshape((3,3))
    # H2to1/=H2to1[2,2]
    return H2to1


def computeH_norm(x1, x2):
 	#Q2.2.2
 	#Compute the centroid of the points
    x1_c = (np.mean(x1[:,0]), np.mean(x1[:,1]))
    x2_c = (np.mean(x2[:,0]), np.mean(x2[:,1]))
    

 	#Shift the origin of the points to the centroid
    x1_bar = np.asarray([[(el[0]-x1_c[0]),(el[1] - x1_c[1])] for el in x1])
    x2_bar = np.asarray([[(el[0]-x2_c[0]),(el[1] - x2_c[1])] for el in x2])

 	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max_x1 = list()
    max_x2 = list()
    
    for i in range(x1_bar.shape[0]):
        max_x1.append(np.sqrt((x1_bar[i, 0])**2 + (x1_bar[i, 1])**2))
        max_x2.append(np.sqrt((x2_bar[i, 0])**2 + (x2_bar[i, 1])**2))
    
    
    #milarity transform 1
    s_x1 = np.sqrt(2)/np.amax(max_x1)
    s_x2 = np.sqrt(2)/np.amax(max_x2)
     
    T1 = np.array([[s_x1, 0, -s_x1*x1_c[0]], [0, s_x1, -s_x1*x1_c[1]], [0, 0, 1]])
    T2 = np.array([[s_x2, 0, -s_x2*x2_c[0]], [0, s_x2, -s_x2*x2_c[1]], [0, 0, 1]])
    x1_new = x1_bar*s_x1
    x2_new = x2_bar*s_x2
    
    H = computeH(x1_new, x2_new)
     
    #Denormalization
    H2to1 = np.linalg.inv(T1) @ (H @ T2)
        
    return H2to1


def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    maxInliers = -np.inf
    p1 = np.vstack((locs1.T, np.ones(locs1.shape[0])))
    p2 = np.vstack((locs2.T, np.ones(locs2.shape[0])))

    for i in range(0, max_iters):
        idx = np.random.choice(locs1.shape[0], 4)
        rand1 = locs1[idx,:]
        rand2 = locs2[idx,:]
        H = computeH_norm(rand1, rand2)
    
        x1_pred = H @ p2
        x1_pred /= x1_pred[2,:]
        
        in_pts = np.zeros(locs1.shape[0])
        
        for i in range(len(in_pts)):
            actual_diff = np.sqrt((p1[0,i] - x1_pred[0,i])**2 + (p1[1,i] - x1_pred[1,i])**2)
            if actual_diff < inlier_tol:
                in_pts[i] = 1
        numInliers = sum(in_pts)
        
        if numInliers > maxInliers:
            maxInliers = numInliers
            inliers = in_pts
            bestH2to1 = H
    
    return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.

	#Create mask of same size as template

    mask = np.ones(template.shape)
    template = cv2.transpose(template)
    mask = cv2.transpose(mask)
	#Warp mask by appropriate homography
    h_mask = cv2.warpPerspective(mask, np.linalg.inv(H2to1), (img.shape[0], img.shape[1]))

	#Warp template by appropriate homography

    h_mask = cv2.transpose(h_mask)
    idx = np.nonzero(h_mask)
    
    new_template = cv2.warpPerspective(template, np.linalg.inv(H2to1), (img.shape[0], img.shape[1]))

	#Use mask to combine the warped template and the image
    new_template = cv2.transpose(new_template)
    
    img[idx] = new_template[idx]
    composite_img = img
    return composite_img



"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here

import helper
import numpy as np
import scipy
import scipy.ndimage as nd

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # pass
    pts1_norm = pts1/M
    pts2_norm = pts2/M
    
    pts1_x, pts1_y = pts1_norm[:,0], pts1_norm[:,1]
    pts2_x, pts2_y = pts2_norm[:,0], pts2_norm[:,1]

    solve_A = np.ones([np.size(pts1_x),9])
    
    solve_A[:,0] = np.multiply(pts2_x, pts1_x)
    solve_A[:,1] = np.multiply(pts2_x, pts1_y)
    solve_A[:,2] = pts2_x
    solve_A[:,3] = np.multiply(pts2_y, pts1_x)
    solve_A[:,4] = np.multiply(pts2_y, pts1_y)
    solve_A[:,5] = pts2_y
    solve_A[:,6] = pts1_x
    solve_A[:,7] = pts1_y
    
    U,S,Vt = np.linalg.svd(solve_A)
    V = np.transpose(Vt)
    F_svd1 = np.reshape(V[:,-1], (3,3))
    F_singular = helper._singularize(F_svd1)
    
    refined_F = helper.refineF(F_singular, pts1_norm, pts2_norm)
    
    T = np.array([[1/M, 0, 0],[0, 1/M, 0], [0,0,1]])
    result = T @ refined_F @ T
    return result


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pass


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    # pass
    result = K2.T @ F @ K1
    return result


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    # pass
    pts1_x = pts1[:,0]
    pts1_y = pts1[:,1]

    pts2_x = pts2[:,0]
    pts2_y = pts2[:,1]

    P = np.zeros((pts1_x.size, 3))
    
    for idx in range(pts1_x.size) :
        solve_A = np.zeros((4,4))
        
        solve_A[0,:] = pts1_x[idx] * C1[2,:] - C1[0,:]
        solve_A[1,:] = pts1_y[idx] * C1[2,:] - C1[1,:]
        solve_A[2,:] = pts2_x[idx] * C2[2,:] - C2[0,:]
        solve_A[3,:] = pts2_y[idx] * C2[2,:] - C2[1,:]
        U,S,Vt = np.linalg.svd(solve_A)

        V = np.transpose(Vt)
        slct = V[:,-1]
        slct /= slct[-1]
        P[idx,:] = slct[0:3]
    
    hold_pts = np.vstack((P.T, np.ones((1, pts1_x.size))))
    
    new_pts1 = C1 @ hold_pts
    new_pts1 /= new_pts1[2,:]
    new_pts1 = new_pts1[:2,:]
    
    new_pts2 = C2 @ hold_pts
    new_pts2 /= new_pts2[2,:]
    new_pts2 = new_pts2[:2,:]
    
    err = np.sum((pts1.T - new_pts1)**2 + (pts2.T - new_pts2)**2)
    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    # pass
    x1 = int(x1)
    y1 = int(y1)
    win_size = 10
    im1_patch = im1[(y1 - win_size//2): (y1 + win_size//2 + 1), (x1 - win_size//2): (x1 + win_size//2 + 1),:]

    pt1 = np.array([x1, y1, 1]) # homogeneous coordinates of im1

    line = F @ pt1

    out_line = line / np.linalg.norm(line)

    pos_y2 = np.array(range(y1-(win_size//2)*8,y1+ (win_size//2)*8))
    pos_x2 = np.round((-out_line[2]-out_line[1]*pos_y2)/out_line[0]).astype(np.int)

    cond = (pos_x2 >= win_size//2) & (pos_x2 + win_size//2 < im2.shape[1]) & (pos_y2 >=win_size//2) & (pos_y2 + win_size//2 < im2.shape[0])
    x2, y2 = pos_x2[cond], pos_y2[cond]
    err_val = np.inf

    for ind in range(len(x2)):
        im2_patch = im2[y2[ind]-win_size//2: y2[ind] + win_size//2+1,x2[ind] -win_size//2: x2[ind] + win_size//2+1,:]
        error = np.linalg.norm((im1_patch-im2_patch))
        gauss_err = np.sum(nd.filters.gaussian_filter(error, sigma=2.0))
        
        if gauss_err < err_val:
                err_val = gauss_err
                result = x2[ind], y2[ind]

    return result

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters, tol):
    # Replace pass by your implementation
    # pass
    max_in = -1
    
    pts1_ = np.vstack((pts1.T,np.ones([1,pts1.shape[0]])))
    pts2_ = np.vstack((pts2.T,np.ones([1,pts1.shape[0]])))

    for i in range(nIters):
        print(i)
        rand_idx = np.random.choice(pts1.shape[0],8)

        rand_pts1 = pts1[rand_idx,:]
        rand_pts2 = pts2[rand_idx,:]
        
        F = eightpoint(rand_pts1, rand_pts2, M)

        pred_x2_ = np.dot(F,pts1_)
        pred_x2  = pred_x2_ / np.sqrt(np.sum(pred_x2_[:2,:]**2, axis=0))

        err = abs(np.sum(pts2_ * pred_x2, axis=0))
        inliers_num = err < tol
        tot_inliers = inliers_num[inliers_num.T].shape[0]
        
        if tot_inliers > max_in:
            F_out = F
            max_in = tot_inliers
            inliers = inliers_num

    return F_out, inliers

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    # pass    
    theta = np.linalg.norm(r)
    
    if theta == 0 :
        R = np.eye(3)
    else :
        u = r / theta
        temp = np.zeros((3,3))
        temp[0,1] = -u[2]
        temp[1,0] = u[2]
        temp[0,2] = u[1]
        temp[2,0] = -u[1]
        temp[1,2] = -u[0]
        temp[2,1] = u[0]
        R = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * (u @ u.T) + np.sin(theta) * temp
    return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    # pass
    theta = np.arccos((np.trace(R) - 1) / 2)
    
    if theta < 0.0000001 :
        return np.zeros((3,1))
    
    mat = np.array([[R[2,1] - R[1,2]], [R[0,2] - R[2,0]], [R[1,0] - R[0,1]]])
    r = (1.0 / (2 * np.sin(theta)) * mat) * theta
    
    return r

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    # pass
    n_pts = x.size // 3 - 2
    pts3d = x[0 : n_pts*3]
    pts3d = np.reshape(pts3d, (pts3d.size // 3, 3))
    ext_vec = x[n_pts*3: n_pts*3 + 3]
    ext_vec = np.reshape(ext_vec, (ext_vec.size,1))
    t = x[n_pts*3 + 3:]

    mat_r = rodrigues(ext_vec)
    t = np.reshape(t, (3,1))
    M2 = np.append(mat_r, t, axis = 1)
    C1 = np.matmul(K1,M1)
    C2 = np.matmul(K2, M2)
    error = np.array([])

    for index in range(0, n_pts) :
        p_3d = pts3d[index,:]
        p_4d = np.append(p_3d, np.array([1]))
        proj_cam1 = np.matmul(C1, p_4d)
        proj_cam2 = np.matmul(C2, p_4d)
        proj_cam1_2D = proj_cam1 / proj_cam1[-1]
        proj_cam2_2D = proj_cam2 / proj_cam2[-1]
        proj_cam1_2D = proj_cam1_2D[0:-1]
        proj_cam2_2D = proj_cam2_2D[0:-1]
        proj_cam1_g = p1[index]
        proj_cam2_g = p2[index]
        error1 = proj_cam1_g - proj_cam1_2D
        error2 = proj_cam2_g - proj_cam2_2D
        error1 = np.reshape(error1, (error1.size,1))
        error2 = np.reshape(error2, (error2.size,1))
        tol_err = np.append(error1, error2, axis = 0)
        if error.size == 0 :
            error = tol_err
        else :
            error = np.append(error, tol_err, axis = 0)
    return error

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    # pass
    n_pts = P_init.shape[0]
    r_err = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x).flatten()
    vec_init = np.zeros((3*n_pts + 6), dtype = float)
    p_init_flatten = np.reshape(P_init, (P_init.size))
    vec_init[0:-6] = p_init_flatten
    mat_r = M2_init[:,0:3]
    ext_vec = invRodrigues(mat_r)
    ext_vec = np.reshape(ext_vec, (ext_vec.size))
    vec_t = M2_init[:,3]
    vec_init[-6:-3] = ext_vec
    vec_init[-3:] = vec_t
    
    R2_0 = M2_init[:, 0:3]
    t2_0 = M2_init[:, 3]
    r2_0 = invRodrigues(R2_0)
    x0 = P_init.flatten()
    x0 = np.append(x0, r2_0.flatten())
    x0 = np.append(x0, t2_0.flatten())
    err_opt = rodriguesResidual(K1, M1, p1, K2, p2, x0)
    err_opt = sum(err_opt**2)
    print('orginial error: '  + str(err_opt))
    
    final_out,_ = scipy.optimize.leastsq(r_err, vec_init)

    rFinal = final_out[n_pts*3 : n_pts*3+3]
    tFinal = final_out[n_pts*3+3:]
    pts3d = final_out[0 : n_pts*3]
    pts3d = np.reshape(pts3d, (pts3d.size // 3, 3))
    ext_vec = final_out[n_pts*3: n_pts*3 + 3]
    ext_vec = np.reshape(ext_vec, (ext_vec.size,1))

    M2_out = np.zeros((3,4))
    M2_out[:,0:3] = rodrigues(rFinal)
    M2_out[:,3] = tFinal
    
    err_opt = rodriguesResidual(K1, M1, p1, K2, p2, final_out)
    err_opt = sum(err_opt**2)
    print('Final error: '  + str(err_opt))
    
    return M2_out, pts3d

'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres):
    # Replace pass by your implementation
    # pass
    # P, err = triangulate(C1, pts1[:, :2], C2, pts2[:, :2])
    pts1 = pts1[pts1[:,2] > Thres]
    p1 = pts1[:,:2]
    pts2 = pts2[pts2[:,2] > Thres]
    p2 = pts2[:,:2]
    pts3 = pts3[pts3[:,2] > Thres]
    p3 = pts3[:,:2]
    
    n, temp = p1.shape
    P = np.zeros((n, 3))
    P_hom = np.zeros((n,4))
    for i in range(n):
        x1, y1 = p1[i,0], p1[i,1]
        x2, y2 = p2[i,0], p2[i,1]
        x3, y3 = p3[i,0], p3[i,1]
        
        A1 = x1*C1[2,:] - C1[0,:]
        A2 = y1*C1[2,:] - C1[1,:]
        A3 = x2*C2[2,:] - C2[0,:]
        A4 = y2*C2[2,:] - C2[1,:]
        A5 = x3*C3[2,:] - C3[0,:]
        A6 = y3*C3[2,:] - C3[1,:]
        A = np.vstack((A1, A2, A3, A4, A5, A6))
        
        u,s, vh = np.linalg.svd(A)
        
        p = vh[-1, :]
        p /= p[3]
        P[i,:] = p[0:3]
        P_hom[i,:] = p
        
    p1_proj = np.matmul(C1, P_hom.T)
    lam1 = p1_proj[-1,:]
    p1_proj /= lam1
    
    p2_proj = np.matmul(C2, P_hom.T)
    lam2 = p2_proj[-1,:]
    p1_proj /= lam2
    
    err1 = np.sum((p1_proj[[0,1], :].T - p1)**2)
    err2 = np.sum((p2_proj[[0,1], :].T - p2)**2)
    err = np.sqrt(err1+err2)
    
    return P, err

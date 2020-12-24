import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Covert 4x4 homogeneous pose matrix to x, y, z, roll, pitch, yaw."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Covert x, y, z, roll, pitch, yaw to 4x4 homogeneous pose matrix."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T

def pose_estimate_nls(K, Twc_guess, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    K          - 3x3 camera intrinsic calibration matrix.
    Twc_guess  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts       - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts       - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array (float64), pose matrix, camera pose in target frame.
    """
    maxIters = 250  # Set maximum iterations.

    tp = Ipts.shape[1]   # Num points.

    J  = np.zeros((2*tp, 6))  # Jacobian matrix to populate later (96 by 6).
    #Convert initial camera pose rotation matrix into initial camera pose estimates (6 x 1).
    x = epose_from_hpose(Twc_guess)
    #Convert World points into a homogeneous matrix (4by4)
    Wpts_h = np.vstack((Wpts, np.ones(tp)))
    #Convert intrinct matrix into homogeneous (4by4 - slide 11, lecture 4)
    K_h = np.vstack((np.hstack((K, np.array([[0,0,0]]).T)), np.array([[0, 0, 0, 1]])))

    iter = 1

    # 2. Main loop - continue until convergence or maxIters.
    while True:    
        #Get predicted x,y coordinates of image plane. Store in np.array "y"
        y = K_h.dot(inv(Twc_guess)).dot(Wpts_h)
        y = y/y[2]
        y = y[0:2,:]
        
        #Get error (in x,y coordinates) between predicated image plane and ground truth image plane coordinates.
        dy = y - Ipts
        #Put errors in column vector format
        dy = np.reshape(dy, (2*tp, 1), 'F')
        
        #Calculate Jacobian
        for i in range(tp):
            #Calculate Jacobian Ji (2 by 6). I.e for each x,y point
            Ji = find_jacobian(K, Twc_guess, Wpts[:,i].reshape(3,1))
            #Stack Ji computed into overall Jacobian Matrix J  for all 48 points (i.e 48*2 by 6 matrix)
            J[2*i:2*i+2,:] = Ji
            
        #Solve for delta x vector (i.e change in camera pose, tx, ty, tz, roll, pitch, yaw)
        dx = -inv(J.T.dot(J)).dot(J.T).dot(dy)
        
        #Calculate new predicted camera pose estimate
        x_new = x + dx   
        
        #Create new Matrix Twc with updated camera pose parameters x
        Twc_guess = hpose_from_epose(x_new)
        
        # Calculate norm between new predicted camera pose estimates and previous camera pose estimates
        diff = norm(x_new - x)

        #Threshold was changed to <1e-6 as solution converged after 5 iterations with this threshold
        if norm(diff) < 1e-6:
            print("Covergence required %d iters." % iter)
            break
        elif iter == maxIters:
            print("Failed to converge after %d iters." % iter)
            break
        
        # Set previous camera pose estimate to latest estimate
        x = x_new
        
        #Increase iteration and repeat procedure from beginning of while loop
        iter += 1
                   
        
    #Set Twc matrix to final Twc_guess after interation is complete
    Twc = Twc_guess

    correct = isinstance(Twc, np.ndarray) and \
        Twc.dtype == np.float64 and \
        Twc.shape == (4, 4) and Twc[3, 3] == 1.0000

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Twc
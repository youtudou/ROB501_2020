import numpy as np
from numpy.linalg import inv
from rpy_from_dcm import rpy_from_dcm
from dcm_from_rpy import dcm_from_rpy
from estimate_motion_ls import estimate_motion_ls

def estimate_motion_ils(Pi, Pf, Si, Sf, iters):
    """
    Estimate motion from 3D correspondences.
  
    The function estimates the 6-DOF motion of a body, given a series 
    of 3D point correspondences. This method relies on NLS.
    
    Arrays Pi and Pf store corresponding landmark points before and after 
    a change in pose.  Covariance matrices for the points are stored in Si 
    and Sf. All arrays should contain float64 values.

    Parameters:
    -----------
    Pi  - 3xn np.array of points (intial - before motion).
    Pf  - 3xn np.array of points (final - after motion).
    Si  - 3x3xn np.array of landmark covariance matrices.
    Sf  - 3x3xn np.array of landmark covariance matrices.

    Outputs:
    --------
    Tfi  - 4x4 np.array, homogeneous transform matrix, frame 'i' to frame 'f'.
    """

    
    tp = Pi.shape[1]                      # Num points.
    
    J  = np.zeros((3*tp, 6))                # Jacobian.
    
    # Initial guess...
    Tfi_guess = estimate_motion_ls(Pi, Pf, Si, Sf)
    
    #Convert initial guess to parameter vector (6 x 1).
    x = epose_from_hpose(Tfi_guess)
    
    iter = 1
    
    # 2. Main loop - continue until convergence or maxIters.
    while True:
        
        #Get predicted 3D world points in frame 'f'. Store in np.array "y"
        y = Tfi_guess[:3,:].dot(np.insert(Pi,3,1, axis =0))
    
        
        #Get error between predicated world points in frame f and ground truth world points in frame f.
        dy = y - Pf
        
        for i in range(tp):
            #calculate matrix weight Sigma_j as per equation given in slide 18, lecture #13
            Sj = inv(Sf[:,:,i] + Tfi_guess[:3,:3].dot(Si[:,:,i]).dot(Tfi_guess[:3,:3].T))
            #multiply errors by matrix weight Sigma_j
            dy = Sj.dot(dy)
        
        #Make dy a column vector consiting of (x1, y1, z1, x2, y2,z2... x_n, y_n, z_n)
        dy = np.reshape(dy, (3*tp, 1), 'F')
        
        #Calculate Jacobian
        for i in range(tp):
            #Calculate Jacobian Ji (3 by 6). I.e for each x,y points
            Ji = find_jacobian(Tfi_guess, Pi[:,i].reshape(3,1))
            #Stack Ji computed into overall Jacobian Matrix J
            J[3*i:3*i+3,:] = Ji
            
        #Solve for delta x vector (i.e change in camera pose, tx, ty, tz, roll, pitch, yaw)
        dx = -inv(J.T.dot(J)).dot(J.T).dot(dy)
        
        # If change in x (dx) is small, stop iteration
        norm_x = np.sqrt(x[0][0]**2+x[1][0]**2+x[2][0]**2+x[3][0]**2+x[4][0]**2+x[5][0]**2)
        norm_dx = np.sqrt(dx[0][0]**2+dx[1][0]**2+dx[2][0]**2+dx[3][0]**2+dx[4][0]**2+dx[5][0]**2)
        if abs(norm_dx/norm_x) < 1e-6:
            print("Covergence required %d iters." % iter)
            break
        #If max interation count is reached, stop interation
        elif iter == iters:
            print("Failed to converge after %d iters." % iter)
            break
        #If not, iterate again
        iter += 1
           
        #Update new predicted camera pose estimate
        x = x + dx
        
        #Create new Matrix Tfi with updated camera pose parameters x
        Tfi_guess = hpose_from_epose(x)
        
    #Set Tfi matrix to final Tfi_guess after interation is complete
    Tfi = Tfi_guess

    # Check for correct outputs...
    correct = isinstance(Tfi, np.ndarray) and Tfi.shape == (4, 4)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Tfi

#This function accepts world points in frame i and
#outputs them in frame f
def f_x(pose_par, Pi):
    
    Tfi = hpose_from_epose(pose_par) #compute homogeneous transform matrix
    Pf = Tfi[:3,:].dot(np.insert(Pi,3,1, axis =0)) #convert points from frame i to frame f
  
    Pf = Pf.reshape(3,)
    return Pf

#This function is given in the assignment
def hpose_from_epose(E):
    """Covert x, y, z, roll, pitch, yaw to 4x4 homogeneous pose matrix."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T

#This function is given in the assignment
def epose_from_hpose(T):
    """Covert 4x4 homogeneous pose matrix to x, y, z, roll, pitch, yaw."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E



def find_jacobian(Tfi_guess,Pi):

    
    #Appromimate Jacobian: partial_f/partial_xi = f(x + epsilon*ui) - f(x) / epsilon
    
    #Epsilon
    e = 0.000001
    
    #Use functions provided in Autolab which converts rotation matrix to camera poses: tx, ty, tz, roll, pitch, yaw
    pose_par = epose_from_hpose(Tfi_guess)
    
    #Define empty Jacobian Matrix
    J = np.zeros((3,6))
    
  
    #Function f_x is defined above. It accepts world points in frame i and returns them in frame f
    # This function is used to obtain the the columns of the Jacobian matrix using the
    # Jacobian approximation: partial_f/partial_xi = f(x + epsilon*ui) - f(x) / epsilon
    
    
    #Computes first column on Jacobian - change in f with respect to tx (since epsilon "disturbance" is introduced in tx)
    J[:,0] = (f_x(pose_par + np.array([[e],[0],[0],[0],[0],[0]]), Pi) - f_x(pose_par, Pi))/e
    #Computes first column on Jacobian - change in f with respect to ty (since epsilon "disturbance" is introduced in ty)
    J[:,1] = (f_x(pose_par + np.array([[0],[e],[0],[0],[0],[0]]), Pi) - f_x(pose_par, Pi))/e
    #Computes first column on Jacobian - change in f with respect to tz (since epsilon "disturbance" is introduced in tz)
    J[:,2] = (f_x(pose_par + np.array([[0],[0],[e],[0],[0],[0]]), Pi) - f_x(pose_par, Pi))/e
    #Computes first column on Jacobian - change in f with respect to roll (since epsilon "disturbance" is introduced in roll)
    J[:,3] = (f_x(pose_par + np.array([[0],[0],[0],[e],[0],[0]]), Pi) - f_x(pose_par, Pi))/e
    #Computes first column on Jacobian - change in f with respect to pitch (since epsilon "disturbance" is introduced in pitch)
    J[:,4] = (f_x(pose_par + np.array([[0],[0],[0],[0],[e],[0]]), Pi) - f_x(pose_par, Pi))/e
    #Computes first column on Jacobian - change in f with respect to yaw (since epsilon "disturbance" is introduced in yaw)
    J[:,5] = (f_x(pose_par + np.array([[0],[0],[0],[0],[0],[e]]), Pi) - f_x(pose_par, Pi))/e
    

    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (3, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J

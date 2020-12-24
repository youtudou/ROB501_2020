import numpy as np
from numpy.linalg import inv

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Note that the homogeneous transformation matrix provided defines the
    transformation from the *camera frame* to the *world frame* (to 
    project into the image, you would need to invert this matrix).

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
         The Jacobian must contain float64 values.
    """
    
    
    #Appromimate Jacobian: partial_f/partial_xi = f(x + epsilon*ui) - f(x) / epsilon
    
    #Epsilon
    e = 0.000001
    
    #Define empty Jacobian Matrix
    J = np.zeros((2,6))
    
    #Use functions provided in Autolab which converts rotation matrix to camera poses: tx, ty, tz, roll, pitch, yaw
    pose_par = epose_from_hpose(Twc)
    
    #Function f_x is defined below. It is the forward camera model. It accepts world points, camera intrinsinct matrix, camera pose
    # and outputs the corresponding image place coordinates.
    # this function is used to obtain the the columns of the Jacobian matrix using the
    # Jacobian approximation: partial_f/partial_xi = f(x + epsilon*ui) - f(x) / epsilon
    
    
    #Computes first column on Jacobian - change in f with respect to tx (since epsilon "disturbance" is introduced in tx)
    J[:,0] = (f_x(pose_par + np.array([[e],[0],[0],[0],[0],[0]]), Wpt, K) - f_x(pose_par, Wpt, K))/e
    #Computes first column on Jacobian - change in f with respect to ty (since epsilon "disturbance" is introduced in ty)
    J[:,1] = (f_x(pose_par + np.array([[0],[e],[0],[0],[0],[0]]), Wpt, K) - f_x(pose_par, Wpt, K))/e
    #Computes first column on Jacobian - change in f with respect to tz (since epsilon "disturbance" is introduced in tz)
    J[:,2] = (f_x(pose_par + np.array([[0],[0],[e],[0],[0],[0]]), Wpt, K) - f_x(pose_par, Wpt, K))/e
    #Computes first column on Jacobian - change in f with respect to roll (since epsilon "disturbance" is introduced in roll)
    J[:,3] = (f_x(pose_par + np.array([[0],[0],[0],[e],[0],[0]]), Wpt, K) - f_x(pose_par, Wpt, K))/e
    #Computes first column on Jacobian - change in f with respect to pitch (since epsilon "disturbance" is introduced in pitch)
    J[:,4] = (f_x(pose_par + np.array([[0],[0],[0],[0],[e],[0]]), Wpt, K) - f_x(pose_par, Wpt, K))/e
    #Computes first column on Jacobian - change in f with respect to yaw (since epsilon "disturbance" is introduced in yaw)
    J[:,5] = (f_x(pose_par + np.array([[0],[0],[0],[0],[0],[e]]), Wpt, K) - f_x(pose_par, Wpt, K))/e
    

    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J

#This function is the forward camera model. It accepts world points, camera intrinsinct matrix, camera pose
#and outputs the corresponding image place coordinates.
def f_x(pose_par, Wpt, K):
    
    """
    Inputs:
    pos_par is a column vector consisting of tx, ty, tz, roll, pitch, yaw
    Wpt is a column vector consisting of X,Y,Z world coordinates
    K is the instrinct matrix of dimenstion 3by3
    
    Outputs:
    x, y image plane coordinates. Dimension (2,)
    """    
    #Convert intrinct matrix into homogeneous (4by4 - slide 11, lecture 4)
    K_h = np.vstack((np.hstack((K, np.array([[0,0,0]]).T)), np.array([[0, 0, 0, 1]])))
    #Calculate homogeneous world points (4by1 column vector)
    Wpt_h = np.vstack((Wpt, np.array([[1]])))
    #Obtain rotation matrix from pose parameters
    Twc= hpose_from_epose(pose_par)
    
    #Calculate corresponding x,y point in the image
    y = K_h.dot(inv(Twc)).dot(Wpt_h)
    y = y/y[2]
    y = y[0:2].reshape(2,)
    
    return y

#This function is given in the assignment
def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

#This function is given in the assignment
def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.  

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])

#This function is given in the assignment
def epose_from_hpose(T):
    """Covert 4x4 homogeneous pose matrix to x, y, z, roll, pitch, yaw."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

#This function is given in the assignment
def hpose_from_epose(E):
    """Covert x, y, z, roll, pitch, yaw to 4x4 homogeneous pose matrix."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T
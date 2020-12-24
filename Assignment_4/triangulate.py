import numpy as np
from numpy.linalg import inv, norm

def triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr):
    """
    Triangulate 3D point position from camera projections.

    The function computes the 3D position of a point landmark from the 
    projection of the point into two camera images separated by a known
    baseline. All arrays should contain float64 values.

    Parameters:
    -----------
    Kl   - 3x3 np.array, left camera intrinsic calibration matrix.
    Kr   - 3x3 np.array, right camera intrinsic calibration matrix.
    Twl  - 4x4 np.array, homogeneous pose, left camera in world frame.
    Twr  - 4x4 np.array, homogeneous pose, right camera in world frame.
    pl   - 2x1 np.array, point in left camera image.
    pr   - 2x1 np.array, point in right camera image.
    Sl   - 2x2 np.array, left image point covariance matrix.
    Sr   - 2x2 np.array, right image point covariance matrix.

    Returns:
    --------
    Pl  - 3x1 np.array, closest point on ray from left camera  (in world frame).
    Pr  - 3x1 np.array, closest point on ray from right camera (in world frame).
    P   - 3x1 np.array, estimated 3D landmark position in the world frame.
    S   - 3x3 np.array, covariance matrix for estimated 3D point.
    """
    #--- FILL ME IN ---
    
    # Compute baseline (right camera translation minus left camera translation)
    Cr = (Twr)[0:3,-1] #left camera translaton
    Cl = (Twl)[0:3,-1] #right camera translation
    b = (Cr - Cl).reshape(3,1)
    
    
    # Unit vectors projecting from optical center to image plane points.
    # Use variables rayl and rayr for the rays.
    rayl = Twl[0:3,0:3].dot(inv(Kl)).dot(np.insert(pl,2,1, axis =0))
    rayl = rayl/norm(rayl) #convert to unit vector
    
    rayr = Twr[0:3,0:3].dot(inv(Kr)).dot(np.insert(pr,2,1, axis =0))
    rayr = rayr/norm(rayr) #convert to unit vector
    
    
    # Projected segment lengths.
    # Use variables ml and mr for the segment lengths.
    rLrR = rayl.T.dot(rayr)[0][0]
    ml = ((b.T.dot(rayl) - (b.T.dot(rayr))*(rLrR))/(1-rLrR**2))[0][0]
    mr = (rLrR*ml - b.T.dot(rayr))[0][0]
    
    # Segment endpoints.
    # User variables Pl and Pr for the segment endpoints.
    Pl = Cl.reshape(3,1) + rayl*ml
    Pr = Cr.reshape(3,1) + rayr*mr
    
    # Now fill in with appropriate ray Jacobians. These are 
    # 3x4 matrices, but two columns are zeros (because the right
    # ray direction is not affected by the left image point and 
    # vice versa).
    drayl = np.zeros((3, 4))  # Jacobian left ray w.r.t. image points.
    drayr = np.zeros((3, 4))  # Jacobian right ray w.r.t. image points.
    
    # Add code here...
    #rayl = f(x)_l/g(x)_l = r/norm(r). Equation for unit vector provided in the assignment
    #drayl = d/dx[f(x)_l/g(x)_l] = ( d/dx[f(x)_l]*g(x)_l - f(x)_l*d/dx[g(x)_l] / [g(x)_l]^2 )
    #where x is the image plane points in the left camera ul (i.e pl[0][0]), vl (i.e pl[1][0]), 
    #and right camera ur (i.e pr[0][0]), vr (i.e pr[1][0])
    
    #As per equation in the assignment. I.e column vector (c1*u, c2*v, c3)
    fxl = Twl[:3,:3].dot(inv(Kl)).dot(np.array([[pl[0][0]],[pl[1][0]],[1]]))
    
    #f(x)_l = column vector(c1*ul, c2*vl + c3). 
    #Therefore f(x)_l w.r.t u = f(x)l_u = column vector (c1, 0, 0,)
    fxl_u = Twl[:3,:3].dot(inv(Kl)).dot(np.array([[1],[0],[0]]))
    #Therefore f(x)_l w.r.t v = f(x)l_v = column vector (0, c2, 0,)
    fxl_v = Twl[:3,:3].dot(inv(Kl)).dot(np.array([[0],[1],[0]]))
    
    #Same math applied as with f(x)_l shown above - only that it is with the right camera
    fxr = Twr[:3,:3].dot(inv(Kr)).dot(np.array([[pr[0][0]],[pr[1][0]],[1]]))
    fxr_u = Twr[:3,:3].dot(inv(Kr)).dot(np.array([[1],[0],[0]]))
    fxr_v = Twr[:3,:3].dot(inv(Kr)).dot(np.array([[0],[1],[0]]))
    
    #Recall from above that g(x)_l = norm(r)
    gxl = norm(fxl)
    #g(x)_l wrt to u is; u*c1^2/norm(r). Where u*c1^2 = fxl_u.T.dot(fxl)
    # and gxl = norm(r)
    gxl_u = fxl_u.T.dot(fxl)/gxl 
    #g(x)_l wrt to v is; v*c2^2/norm(r). Where v*c2^2 = fxl_v.T.dot(fxl)
    # and gxl = norm(r)   
    gxl_v = fxl_v.T.dot(fxl)/gxl
    
    # same as above except with the right camera
    gxr = norm(fxr)
    gxr_u = fxr_u.T.dot(fxr)/gxr
    gxr_v = fxr_v.T.dot(fxr)/gxr
    
    #Fill in Jacobian results with results from above 
    drayl[:,0] = ((fxl_u.dot(gxl) - fxl.dot(gxl_u))/(gxl*gxl)).reshape(3,)
    drayl[:,1] = ((fxl_v.dot(gxl) - fxl.dot(gxl_v))/(gxl*gxl)).reshape(3,) 
    drayr[:,2] = ((fxr_u.dot(gxr) - fxr.dot(gxr_u))/(gxr*gxr)).reshape(3,)
    drayr[:,3] = ((fxr_v.dot(gxr) - fxr.dot(gxr_v))/(gxr*gxr)).reshape(3,)
    
    
    
    #------------------
    
    # Compute dml and dmr (partials wrt segment lengths).
    # Compute dml and dmr (partials wrt segment lengths).
    u = np.dot(b.T, rayl) - np.dot(b.T, rayr)*np.dot(rayl.T, rayr)
    v = 1 - np.dot(rayl.T, rayr)**2

    du = (b.T@drayl).reshape(1, 4) - \
         (b.T@drayr).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayr)*((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
 
    dv = -2*np.dot(rayl.T, rayr)*((rayr.T@drayl).reshape(1, 4) + \
        (rayl.T@drayr).reshape(1, 4))

    m = np.dot(b.T, rayr) - np.dot(b.T, rayl)@np.dot(rayl.T, rayr)
    n = np.dot(rayl.T, rayr)**2 - 1

    dm = (b.T@drayr).reshape(1, 4) - \
         (b.T@drayl).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayl)@((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
    dn = -dv

    dml = (du*v - u*dv)/v**2
    dmr = (dm*n - m*dn)/n**2

    # Finally, compute Jacobian for P w.r.t. image points.
    JP = (ml*drayl + rayl*dml + mr*drayr + rayr*dmr)/2
    
    #--- FILL ME IN ---
    
    # 3D point.
    P = (Pl + Pr)/2
    
    # 3x3 landmark point covariance matrix (need to form
    # the 4x4 image plane covariance matrix first).
    M = np.zeros((4,4))
    M[0:2,0:2] = Sl
    M[2:4,2:4] = Sr
    
    S = JP.dot(M).dot(JP.T) #as per equation in the assignment

    # Check for correct outputs...
    correct = isinstance(Pl, np.ndarray) and Pl.shape == (3, 1) and \
              isinstance(Pr, np.ndarray) and Pr.shape == (3, 1) and \
              isinstance(P,  np.ndarray) and P.shape  == (3, 1) and \
              isinstance(S,  np.ndarray) and S.shape  == (3, 3)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Pl, Pr, P, S
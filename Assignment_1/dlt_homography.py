import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
    
    #Define array x and y containing the x and y coordinates of Image 1 respectively
    x = I1pts[0]
    y = I1pts[1]

    #Define array u and v containing the x and y coordinates of Image 2 respectively
    u = I2pts[0]
    v = I2pts[1]
    
    #Define an 8 by 9 with 0 values to populate later on 
    A = np.zeros((8,9))
    

    # Creates Matrix: four 2 by 9 Ai matrices (one per point correspondence) 
    # and stacks them on top of one another to create a single 8 by 9 matrix A
    for i in range(len(x)):
        Ai = np.array(  [
                        [-x[i], -y[i],  -1, 0,       0,     0,  u[i]*x[i],  u[i]*y[i], u[i]],
                        [0,      0,      0, -x[i],   -y[i], -1, v[i]*x[i],  v[i]*y[i], v[i]],
                        ])
        A[2*i:2*i+2,:] = Ai
    
    #Use nullspace function to solve A*h = 0    
    h = null_space(A)
    
    #Create final 3by3 homography matrix and normalize by lower right entry
    H = h.reshape(3,3)
    H = H/H[2,2]
    
    #------------------

    return H, A

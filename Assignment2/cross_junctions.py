import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *

# You may add support functions here, if desired.

def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #The 4 points of the bounding polygon provided (ordered clockwise starting at upper left corner) 
    # appear to be on the edge of the checkerboard
    # Use the outer 4 cross junction world points provided (which encircle the checkerboard)
    # and approximate the equivalent world points of the bounding polygon coordinates. 
    #Note cross junction world points are provided in row major order
    
    
    #length of each square specified in assignment
    l_square = 63/1000 
    
    #thickness of checkerboard edge is approximately a third of the length of the checkerboard
    t_edge = l_square*(1/3) 
    
    #Equivalent bounding polygon world points
    UL_corner_w_x = Wpts[0, 0] - l_square - t_edge
    UL_corner_w_y = Wpts[1, 0] - l_square - t_edge
    
    BL_corner_w_x = Wpts[0, 40] - l_square - t_edge
    BL_corner_w_y = Wpts[1, 40] + l_square + t_edge
    
    UR_corner_w_x = Wpts[0, 7] + l_square + t_edge
    UR_corner_w_y = Wpts[1, 7] - l_square - t_edge
    
    BR_corner_w_x = Wpts[0, -1] + l_square + t_edge
    BR_corner_w_y = Wpts[1, -1] + l_square + t_edge
    
    #Put equivalent bounding box world points in array - ordered clockwise starting from upper right corner
    bpoly_w = np.array([ [UL_corner_w_x, UR_corner_w_x, BR_corner_w_x, BL_corner_w_x],
                         [UL_corner_w_y, UR_corner_w_y, BR_corner_w_y, BL_corner_w_y]
                      ])
    
    
    #Use Homography function to get homography H that will convert world coordinates into image coordinates
    H, A = dlt_homography(bpoly_w, bpoly)
    
    #Loop through all world point cross junctions to get equivalent image coordinate cross junctions
    
    # Define numpy array with 0's in which to populate image plane cross junctions later on
    x_junctions = np.zeros((3,Wpts.shape[1]))
    
    #Feeds in all the world point cross junctions, calculates the equivalent x,y cross junction 
    #coordinates in image frame 
    for i in range(Wpts.shape[1]):
        x_junctions[:,i]= H.dot(np.array([Wpts[0,i], Wpts[1,i], 1]))
    x_junctions = x_junctions[0:2]/x_junctions[2]
    
    
    # Obtain image patch with with approximated cross junctions obtained so far and feed patch into
    # saddle points function to obtain a more precise cross junction
    
    # window used to define Image patch
    patch_window = 15
    
    
    for j, i in enumerate(x_junctions.T):
        #Extract x and y coordinates of cross junctions
        x, y = int(round(i[0])), int(round(i[1]))
        #Define image patch
        I_patch = I[y - patch_window : y + patch_window, x - patch_window : x + patch_window]
        #Feed image patch into saddle point function to obtain a more precise cross junction location
        # for that image patch
        pt = saddle_point(I_patch)
        # Updated the x_junctions array with the updated "more precise" cross junction points obtained through the
        # saddle points function. Note - need to convert saddle point coordinates of image patch to saddle point
        #coordinates of the overall image I
        x_junctions[:, j] = np.array([pt[0][0] + x - patch_window, pt[1][0] + y - patch_window])
        
    Ipts = x_junctions

    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts

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


def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #Create Matrices Linear Least Squares: Ax = b
    
    # Define dimenion of Image
    rows, columns = I.shape
    
    #Define dimension of A matrix and b column vector
    A = np.zeros((rows*columns, 6))
    b = np.zeros((rows*columns, 1))
    
    #Populate Matrix A and column vector b as per eqn 4 in Saddle Points paper
    i = 0
    for y in range(rows):
        for x in range(columns):
            #Ai provides the equation for each pixel coordinate
            Ai = np.array([x*x, x*y, y*y, x, y, 1])
            #Each new equation (Ai) will be placed in a new row of Matrix A 
            A[i,:] = Ai 
            #populate column vector with a pixel intensity value for each pixel coordinate
            b[i] = I[y, x] 
            #Increment row counter used in Matrix A and column vector b
            i = i + 1
            
    #Solve for x in Ax = b by using Linear least squares function
    # x in this case will be the coefficients listed in equation 4
    alpha, beta, gamma, delta, epsilon, zeta = lstsq(A, b, rcond = None)[0]   
    
    #Find saddle points by finding intersection of the 2 lines. This equation is found below equation 4
    pt = -inv(np.array([[2*alpha[0], beta[0]],[beta[0], 2*gamma[0]]])).dot(np.array([delta[0],epsilon[0]]))
    pt.reshape(2,1)
    pt = pt.reshape(2,1)
    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt
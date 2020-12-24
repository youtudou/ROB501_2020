import numpy as np
from numpy.linalg import inv, lstsq

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
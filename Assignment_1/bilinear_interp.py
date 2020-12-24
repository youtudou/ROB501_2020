import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    

    #Define x, y, x1, y1, x2, y2 values based on first diagram show in:
    #https://en.wikipedia.org/wiki/Bilinear_interpolation
    x = pt[0][0]
    y = pt[1][0]
        
    x1 = int(round(np.floor(pt[0])[0]))
    x2 = int(round(np.ceil(pt[0])[0]))
    
    y1 = int(round(np.floor(pt[1])[0]))
    y2 = int(round(np.ceil(pt[1])[0]))
    
    #obtain the pixel intesity values at Q11, Q21, Q12, Q22 based on the same wiki link
    f_Q11 = I[y1, x1]
    f_Q21 = I[y1, x2]
    f_Q12 = I[y2, x1]
    f_Q22 = I[y2, x2]
    
    # obtain c coefficients by using "Alternative algorithm" section shown on the same wiki link
    c = inv(np.array([  [1, x1, y1, x1*y1],
                        [1, x1, y2, x1*y2],
                        [1, x2, y1, x2*y1],
                        [1, x2, y2, x2*y2]
                    ])).T.dot(np.array([1, x, y, x*y]))
    
    # find linearly interpolated pixel intensity value based on formula found using 
    # "Alternative algorithm" section shown on the same wiki link
    b = round(c[0] * f_Q11 + c[1] * f_Q12 + c[2] * f_Q21 + c[3] * f_Q22)


    return b

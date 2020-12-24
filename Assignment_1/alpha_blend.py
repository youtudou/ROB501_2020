import numpy as np

def alpha_blend(Ifg, Ibg, alpha):
    """
    Performs alpha blending of two images.

    The alpha parameter controls the amount of blending between the foreground
    image and the background image. The background image is assumed to be fully
    opaque.

    Note that images are (usually) integer-valued functions, therefore the 
    image you return must be integer-valued (use round()).

    This function should accept both greyscale and RGB images, and return an
    image of the same type.

    Parameters:
    -----------
    Ifg    - Greyscale or RGB foreground image, 8-bit np.array (i.e., uint8).
    Ibg    - Greyscale or RGB background image, 8-bit np.array (i.e., uint8).
    alpha  - Floating point blending parameter, [0, 1].

    Returns:
    --------
    Ia  - Alpha-bended image (*must be* same size as original images).
    """

    if Ifg.shape != Ibg.shape:
        raise ValueError('Input images are different sizes.')

    # Check if images are RBG (i.e 3 bands)
    if len(Ifg.shape) == 3:

        #Define dimension of image
        rows, columns, bands = Ibg.shape
        
        # Split image pixels into R, G, B bands
        Ibg_R = Ibg[:,:,0]
        Ibg_G = Ibg[:,:,1]
        Ibg_B = Ibg[:,:,2]
        
        Ifg_R = Ifg[:,:,0]
        Ifg_G = Ifg[:,:,1]
        Ifg_B = Ifg[:,:,2]
        
        # Apply alpha blending formula for each band
        Ia_R = np.round(alpha * Ifg_R + (1 - alpha) * Ibg_R)
        Ia_G = np.round(alpha * Ifg_G + (1 - alpha) * Ibg_G)
        Ia_B = np.round(alpha * Ifg_B + (1 - alpha) * Ibg_B)
        
        #Define numpy array Ia with dimensions equal to Image
        Ia = np.zeros((rows, columns, bands)).astype(np.uint8)
        
        #Populate Ia with blending pixel values on each band
        Ia[:,:,0] = Ia_R 
        Ia[:,:,1] = Ia_G 
        Ia[:,:,2] = Ia_B
        
        
        #------------------
    
        return Ia
    
    # Check if images are greyscale (i.e only 1 band)
    elif len(Ifg.shape) == 2:

        # Apply alpha blending formula for grey scale
        Ia = np.round(alpha * Ifg + (1 - alpha) * Ibg)
        
        return Ia
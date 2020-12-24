import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    #Define disparity Image (Id) size with 0 intensity pixel values. 
    #This Id image array will be filled later on with disparity intensity values
    #Id image should have the same dimentions as Il/Ir image
    Id = np.zeros(Il.shape)
    
    #Get pixel coordinates which will define that range in which to scan for correspondances. 
    #This is provided via the bbox array
    x_min = bbox[0,0]
    x_max = bbox[0,1]
    y_min = bbox[1,0]
    y_max = bbox[1,1]
    
    #window size used to find image patch correspondance
    #I.e - in this case the patch size will consist of 10 by 10 pixels
    window = 10 
    
    #For these stereo images, the images are already recitified, so it is only necessary
    # to scan in the x direction for every y coordinate (i.e 1D disparity search)
    
    #Scan accross every coordinate within the bounding box in "window" intervals
    for y in range(y_min, y_max, window):
        for x in range(x_min, x_max, window):
            
            #Obtain window patch in left image
            Il_patch = Il[y: y+window, x : x+window]
            
            #Creates an empty list in which to populate later on with Sums of Absolute differences for various
            #disparity choices
            SAD_list = []
            
            #Looks for various disparity values by incrementing by 1 pixel each time from the current x coordinate
            for i in range(maxd):
                
                #Ensures that the current x coordinate minus the disparity is not outside the right hand image
                #(I.e x coordinate in the right hand image will stay be >0)
                if x-i < 0: 
                    break
                
                #Finds the equivalent window patch in the righ hand image for the various disparities
                Ir_patch = Ir[y:y + window, x-i:x-i+window]
                
                #Computes the sum of absolute differences between the left hand image patch and the
                # right hand image patch: SUM(|Ileft(x,y) - Iright(x-d,y)|)
                SAD = sum(abs(Il_patch.flatten() - Ir_patch.flatten()))
                
                #Stores the currently computed SAD value in a list
                SAD_list.append(SAD)
    
            #Looks through the various SAD values that are stored in the SAD_list, picks the one that yielded the 
            #lowest SAD value and extracts the corresponding disparity value 
            d = SAD_list.index(min(SAD_list))
            
            #Sets the image patch equal to the disparity values in the disparity image map
            Id[y:y+window, x:x+window] = d

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id
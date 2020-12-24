import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

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


# ------------------------------------------- Methodology used -----------------------------------------------------------
  #Census Transform algorithm is used which turns the pixels within the patch into a bit array
  #which indicates if each pixel in the patch is darker or lighter compared to the middle pixel in the patch.
  #The disparity value is chosen based on the patch location in the right-hand image that has the closest bit array match
  # to the left hand image patch.
    
  #Example:
  
    #Left image patch pixel values:
    #40, 45, 15
    #30, 50, 60
    #25, 80, 90
    # Cencus transform for left image patch (CT_L)= 0,0,0,0,1,0,1,1 (row major order)
    
    #Right image patch pixel values:
    #80, 45, 15
    #30, 60, 90
    #25, 80, 10
    # Census transform for right image patch (CT_R)= 1,0,0,0,1,0,1,0 (row major order)
    
    #Census tranform difference(difference in bits between CT_L and CT_R) =  2
    
#----------------------------------------------------------------------------------------------------------------------------    
       
    
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
    

    def cencus_transform(I_patch, window):
        """
        This function returns a bit array indicating if each pixel within the image patch is darker or lighter
        compared to the middle pixel in the patch. If the pixel is darker, then the pixel is converted to a 0 and if
        the pixel is lighter, then it is converted to a 1
        
        Inputs: Image patch, and window size (note - window size needs to be odd (i.e 3 by 3, 5 by 5 etc.))
        Outputs: Bit array (1D) in row major order. Note - the bit string excludes the middle pixel. 
                I.e if a 3 by 3 image patch is the input (9 pixel values), 
                then the output will be an 8 bit string (removes the middle pixel)
        """        
        if (window%2 == 0): #Raises error if window size is not an odd number
            raise TypeError("Use an odd number for window size")
        
        #Extracts middle pixel value and index from Image patch
        middle_pixel_index = int((len(I_patch.flatten())-1)/2)
        middle_pixel_value = I_patch.flatten()[middle_pixel_index]
        
        #Removes the middle image patch pixel and creates a 1D array (in row major order) 
        patch_array = np.delete(I_patch.flatten(), middle_pixel_index)
        
        binary_array = [] #Creates an empty list in which to populate later with a bit string
        
        #Compares each pixel value to the middle pixel in the image patch.
        #Assigns a value of 0 if pixel value <= middle pixel value
        #Assigns a value of 1 is pixel value > middle pixel value
        for val in patch_array:
            if val <= middle_pixel_value:
                binary_array.append(0)
            else:
                binary_array.append(1)
                
        return np.array(binary_array) #Returns bit array as 1D numpy array
            
    def census_transform_difference(binary_array_left, binary_array_right):
        """
        This function compares the left and right image patch bit array and determines how many bits are different 
        Example: 
        Left bit array = [0,0,0,1,1,1,0,0]
        Right bit array =[0,0,0,0,1,1,0,1]
        
        Return = 2 (based on the example bit arrays listed above)
        """
        diff = sum(abs(binary_array_left - binary_array_right))
        return diff
    
    
    
    
    #window size used to find image patch correspondance
    #I.e - in this case the patch size will consist of 7 by 7 pixels    
    window = 7 
    
    #For these stereo images, the images are already recitified, so it is only necessary
    # to scan in the x direction for every y coordinate (i.e 1D disparity search)
    
    #Scan accross every coordinate within the bounding box in "window" intervals
    for y in range(y_min, y_max, window):
        for x in range(x_min, x_max, window):
            
            #Obtain window patch in left image
            Il_patch = Il[y: y+window, x : x+window]
            
            #Obtains binary values (1D array) relative to the middle pixel for the left image patch
            cencus_transform_left = cencus_transform(Il_patch, window)
            
            #Create list that will later be filled with various Census Transform differences between left 
            # and right image patch. I.e Number of binary values that are different between left binary (computed from left image patch)
            #array and right binary array (computed from right image patch)
            CT_list = []
                      
            #Looks for various disparity values by incrementing by 1 pixel each time from the current x coordinate
            for i in range(maxd):
                
                #Ensures that the current x coordinate minus the disparity is not outside the right hand image
                #(I.e x coordinate in the right hand image will stay be >0)
                if x-i < 0:
                    break
                
                #Finds the equivalent window patch in the righ hand image for the various disparities
                Ir_patch = Ir[y:y + window, x-i:x-i+window]
                
                #Obtains binary values (1D array) relative to the middle pixel for the right image patch
                cencus_transform_right = cencus_transform(Ir_patch, window)
                
                #Computes cencus tranform difference (CT difference).
                #I.e Number of binary values that are different between left binary (computed from left image patch)
                #array and right binary array (computed from right image patch)
                CT = census_transform_difference(cencus_transform_left, cencus_transform_right)
                
                #Stores binary difference in a list 
                CT_list.append(CT)
    
            #Looks through the various CT differences values that are stored in the CT_list, picks the one that yielded the 
            #lowest CT difference and extracts the corresponding disparity value 
            d = CT_list.index(min(CT_list))
            
            #Sets the image patch equal to the disparity values in the disparity image map
            Id[y:y+window, x:x+window] = d
    
    Id =  median_filter(Id,20) #applies a median filter to smoothen out outliers.

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id
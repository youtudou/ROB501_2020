# Terrain overlay script file.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from alpha_blend import alpha_blend

def terrain_overlay(Ifg = None, Ibg = None, 
                    Ifg_pts = None, Ibg_pts = None, bbox = None):
    """
    Help rover drivers to safely navigate around obstacles by producing a
    terrain overlay!

    Returns:
    --------
    Iover  - Terrain overlay RGB image, 8-bit np.array (i.e., uint8).
    """
    if not Ifg:
      # Bounding box in rover navcam image (if useful).
      bbox = np.array([[62, 1242, 1242, 62], [190, 190, 794, 794]])
  
      # Point correspondences.
      Ibg_pts = np.array([[410, 928, 1240, 64], [192,  192, 792, 786]])
      Ifg_pts = np.array([[2, 898, 898, 2], [2, 2, 601, 601]])

      Ibg = imread('../images/rover_forward_navcam.png')
      Ifg = imread('../images/false_colour_overlay.png')

    #Creates copy of Ibg image and stores it in array Iover 
    Iover = np.asarray(Ibg).copy()
    
    Ifg = np.asarray(Ifg)
    alpha = 0.7
    
    #Computes homography H
    H, A = dlt_homography(Ibg_pts, Ifg_pts)
    
    #Uses point correspondences in Ibg to define bounding box
    #This bounding box defines the area in which to overlay Ifg onto Ibg
    rover_box = Path(Ibg_pts.T)
    
    #Splits Ifg image into R,G,B bands
    Ifg_R = Ifg[:,:,0]
    Ifg_G = Ifg[:,:,1]
    Ifg_B = Ifg[:,:,2]
    
    #Loops through entire pixel coordinates defined by min and max x range, and min and max y range
    # in Ibg point correspondance
    for x in range(min(Ibg_pts[0]), max(Ibg_pts[0]+1)):
        for y in range(min(Ibg_pts[1]), max(Ibg_pts[1]+1)):
            
            #Skips to next pixel coordinate if not in rover_bouding box defined above
            if rover_box.contains_point(np.array([x,y])) == False:
                continue
           
            #Computes equivalent pixel coordinate in Ifg image 
            x_prime = H.dot(np.array([x,y,1]))
            x_prime = x_prime/x_prime[2]
            x_prime = x_prime[0:2].reshape(2,1)
        
            #Uses equivalent Ifg coordinate to obtain bilinear interpolated Ifg pixel value
            #This is done in all 3 bands
            b_R = bilinear_interp(Ifg_R, x_prime)
            b_G = bilinear_interp(Ifg_G, x_prime)
            b_B = bilinear_interp(Ifg_B, x_prime)
        
            #Replaces pixel value in Ibg image with the bilinearly interpolated pixel value
            # in Ifg image for all three bands
            Iover[y,x,0] =  b_R
            Iover[y,x,1] =  b_G
            Iover[y,x,2] =  b_B
        
    # Iover image is now Ibg image with warped Ifg imaged overlayed onto it.
    # Ibg image is the original rover image (no modification). Uses both images
    # to create a blending "transparency" in warped Ifg area
    Iover = alpha_blend(Iover, Ibg, alpha)


    plt.imshow(Iover)
    plt.show()
    #imwrite('terrain_overlay.png', Iover)

    #------------------

    return Iover

if __name__ == "__main__":
    Iover = terrain_overlay()
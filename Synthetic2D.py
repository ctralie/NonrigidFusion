# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:43:09 2020

@author: Rachel
"""

"""
Purpose: To simulate a 2D range camera by finding the largest
contour on an image and using it as the object of interest
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import skimage
import math
from shapely.geometry import LineString, LinearRing

class FakeScanner2D(object):
    """
    Attributes
    ----------
    I: ndarray(M, N)
        A grayscale image representing a solid object
    contour: ndarray(K, 2)
        A list of points on the boundary of the object
    """

    def __init__(self, path, cutoff=0.9):
        """
        Setup the image that will be used in the fake
        scanner, and extract the boundary contour
        Parameters
        ----------
        path: string
            Path to image file which will be used to simulate a solid object
        cutoff: float
            A number between 0 and 1 used to determine the cutoff at which
            to draw level sets
        """
        I = mpimg.imread(path)
        # Convert to grayscale
        self.I = 0.3*I[:, :, 0] + 0.59*I[:, :, 1] + 0.11*I[:, :, 2]
        contours = skimage.measure.find_contours(self.I, cutoff)
        largest = np.array([[]])
        if len(contours) > 0:
            sizes = np.array([c.shape[0] for c in contours])
            largest = contours[np.argmax(sizes)]
        self.contour = largest

    def display_contour(self):
        """
        Draw the image, along with the contour
        """
        plt.imshow(self.I, cmap='gray')
        plt.plot(self.contour[:, 1], self.contour[:, 0])
    
    def get_range_scan(self, pos, towards, fov, res, do_plot=False):
        """
        Simulate a range scan
        Parameters
        ----------
        pos: ndarray(2)
            The x/y position of the camera
        towards: ndarray(2)
            A unit vector representing where the camera
            is pointed
        fov: float
            The field of view of the camera in radians
        res: int
            The number of samples to take
        do_plot: boolean
            Whether to plot the camera position and direction
        """

        W, H = self.I.shape
        max_dist = 2*np.sqrt(W**2 + H**2)
        ring = LinearRing(np.fliplr(self.contour)) ## TODO: Why fliplr?
        
        ## These two lines are the key lines for finding the intersection
        ## but the direction will change from towards to something else
        ## based on your viewing angle
        line = LineString([pos, pos+towards*max_dist])
        x = line.intersection(ring)
        
        ## Initialize an array that will hold the range scan, and return 
        ## at the end
        range_scan = np.linspace(np.inf, np.inf, res) 
        
        
        if do_plot: ## Eventually, we want to make sure that even if we choose
            ## not to display a debugging plot that shows the intersections, our
            ## code is still able to compute the range scan and return it
            plt.subplot(1, 2, 1) # 1 Row of plots, 2 columns of plots, do the first one
            self.display_contour()
            plt.scatter([pos[0]], [pos[1]])
            
            thetas = np.linspace(-fov/2, fov/2, res) #all thetas within field of view
            
            ## This is the last step of extracting an intersection, so
            ## this is another thing you'll want to do for every ray
            ## This checks if there was an intersection at all
            ## If it doesn't intersect, you can set the range to np.inf (infinity)
            if not x.is_empty:
                # If there was an intersection, draw the closest one
                x = np.array(x[0]) # Extract closest intersection
                segment = np.array([pos, x]) 
                plt.plot(segment[:, 0], segment[:, 1]) 
                plt.scatter(x[0], x[1])
            
            #towards = np.array([pos[0], pos[1], x[0], x[1]])
            #towards = np.array([ x[0], x[1]])

            #right = np.array([pos[0], pos[1],-x[1], x[0]])
            right = np.array([towards[1], -towards[0]])
            
            #plt.plot(right[0::2], right[1::2])
            #plt.scatter(right[2], right[3])
            
            rightdisp = pos + 50*right
            plt.plot([pos[0], rightdisp[0]], [pos[1], rightdisp[1]])
            plt.scatter(rightdisp[0], rightdisp[1])
            
            for i,theta in enumerate(thetas):
                    ## TODO: Update this equation to be in the new coordinate
                    ## system with "towards" and "right" vectors
                    ## Ex) If I had a vector T and a vector R, which were each
                    ## 2-element numpy arrays, I could write V = T + a*R, where
                    ## a is some scalar, and V will then be a 2-element 
                    ## numpy array which you can think of as a vector
                V = towards + math.atan(theta)*right
                plt.plot([pos[0], pos[0]+100*V[0]], [pos[1], pos[1]+100*V[1]])
                
                temp_line = LineString([pos, pos+V*max_dist]) #problem line
                y = temp_line.intersection(ring) #find where these hit contour
                #print(list(temp_line.coords))
                #print(list(y.coords))
                if not y.is_empty: #if so:
                    y = np.array(y[0])
                    plt.scatter(y[0], y[1])
                    diff = y - pos
                    dist = np.sqrt(np.sum(diff**2))
                    range_scan[i] = dist #depth is magnitude?  Yes!
            plt.axis('equal')
            plt.gca().invert_yaxis()
      
        ## Show the range scan in the second subplot on the right
        plt.subplot(1, 2, 2)
        plt.plot(thetas*180/np.pi, range_scan)
        return range_scan
                    
    
"""        
def is_even(x):
    result = False
    if x % 2 == 0:
        result = True
    return result
"""


scanner = FakeScanner2D("fish.png")
pos = np.array([100, 100]) # Position of the camera
towards = np.array([1, 1]) # Direction of the camera
fov = np.pi/2 # Field of view of the camera
res = 100 # Resolution of the camera
scanner.get_range_scan(pos, towards, fov, res, do_plot=True)
plt.show()

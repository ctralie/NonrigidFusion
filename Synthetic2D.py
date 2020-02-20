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
        right = np.array([towards[1], -towards[0]])
        
        if do_plot:
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
            if do_plot:
                plt.plot([pos[0], pos[0]+100*V[0]], [pos[1], pos[1]+100*V[1]])
            
            temp_line = LineString([pos, pos+V*max_dist]) #problem line
            y = temp_line.intersection(ring) #find where these hit contour
            #print(list(temp_line.coords))
            #print(list(y.coords))
            if not y.is_empty: #if so:
                y = np.array(y[0])
                diff = y - pos
                dist = np.sqrt(np.sum(diff**2))
                range_scan[i] = dist #depth is magnitude?  Yes!
                if do_plot:
                    plt.scatter(y[0], y[1])
        
        if do_plot:
            plt.axis('equal')
            plt.gca().invert_yaxis()
            ## Show the range scan in the second subplot on the right
            plt.subplot(1, 2, 2)
            plt.plot(thetas*180/np.pi, range_scan)
        return range_scan


class Reconstruction2D(object):
    """
    Attributes
    ----------
    res: int
        The number of pixels across each axis in the grid
    xmin: float
        Min x coordinate
    xmax: float
        Max x coordinate
    ymin: float
        Min y coordinate
    ymax: float
        Max y coordinate
    X: ndarray(res, res)
        The x coordinates of all pixels in the grid
    Y: ndarray(res, res)
        The y coordinates of all pixels in the grid
    """
    
    def __init__(self, res, xmin, xmax, ymin, ymax):
        # Store grid information as local variables
        self.res = res
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        pixx = np.linspace(xmin, xmax, res)
        pixy = np.linspace(ymin, ymax, res)
        self.X, self.Y = np.meshgrid(pixx, pixy)
        

scanner = FakeScanner2D("fish.png")
pos = np.array([100, 100]) # Position of the camera
towards = np.array([1, 1]) # Direction of the camera
fov = np.pi/2 # Field of view of the camera
res = 100 # Resolution of the camera
range_scan = scanner.get_range_scan(pos, towards, fov, res, do_plot=True)
plt.show()

recon = Reconstruction2D(200, 0, 800, 0, 800)
plt.subplot(121)
plt.imshow(recon.X)
plt.colorbar()
plt.subplot(122)
plt.imshow(recon.Y)
plt.colorbar()
plt.tight_layout()

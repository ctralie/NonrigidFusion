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
from sklearn.neighbors import KDTree
import skimage
import math
import time
from Lines2D import ray_intersect_loop

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
        self.contour = np.flipud(largest)

    def display_contour(self):
        """
        Draw the image, along with the contour
        """
        plt.imshow(self.I.T, cmap='gray')
        plt.plot(self.contour[:, 0], self.contour[:, 1])
    
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
        
        Returns
        -------
        range_scan: ndarray(res)
            The depth scans
        normals: ndarray(res, 2)
            The normals associated to each point of intersection
        """

        W, H = self.I.shape
        
        ## Initialize an array that will hold the range scan, and return 
        ## at the end
        range_scan = np.linspace(np.inf, np.inf, res)
        normals = np.zeros((res, 2))        
        
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
            
            res = ray_intersect_loop(pos, V, self.contour)
            if  res['hit']:
                # Depth is distance of camera to intersection
                y = res['p']
                diff = y - pos
                dist = np.sqrt(np.sum(diff**2))
                range_scan[i] = dist
                normals[i, :] = res['n']
                if do_plot:
                    plt.scatter(y[0], y[1])
                    plt.plot([y[0], y[0]+50*+normals[i, 0]], [y[1], y[1]+50*normals[i, 1]])
        
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
    XGrid: ndarray(res*res, 2)
        The locations of the points on the grid
    SDF: ndarray(res, res)
        The signed distance image
    weights: ndarray(res, res)
        Weights for the SDF points (see more info in Curless/Levoy)
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
        X, Y = np.meshgrid(pixx, pixy)
        self.XGrid = np.array([X.flatten(), Y.flatten()])
       # print(self.XGrid)
        self.SDF = np.inf*np.ones((res, res)) # The signed distance image
        
        
        self.weights = np.ones((res, res))
    
    def incorporate_scan(self, pos, towards, fov, range_scan):
        """
        Given a range scan, update the signed distance image and the weights
        to incorporate the new scan.  You may assume
        Parameters
        ----------
        pos: ndarray(2)
            The x/y position of the camera that took this scan
        towards: ndarray(2)
            A unit vector representing where the camera that took
            this scan was pointed
        fov: float
            The field of view of the camera in radians
        range_scan: ndarray(N)
            The range scan of the scanner
        """

        ## TODO: Transform the range scan into x/y coordinates based on the
        ## position, orientation, and field of view of the camera.  There
        ## are a number of ways to do this, but the easiest way is probably
        ## just to setup a bunch of directions again based on towards and right,
        ## making sure that the final direction vector is a unit vector.  Then,
        ## given a direction *v* and a distance *d*, the position would be
        ## the vector addition p + d*v
        
        right = np.array([towards[1], -towards[0]])
        
        thetas = np.linspace(-fov/2, fov/2, res)
        
        V = np.ones((res,2)) #will hold vectors
        #print(V)        
        
        #recreation of x,y coordinates for each intersection as unit vectors
        for i,theta in enumerate(thetas):
            v = towards + math.atan(theta)*right
            m = v[0] + v[1]
            v = v / m
           # print(v) #unit vector test
            V[i] = pos +( range_scan[i]*v)
            print(V[i])  #test
            
       
       # for x in (self.XGrid):
            #print(self.XGrid[i])
            
            
        tree = KDTree(V)
        distances, indices = tree.query(self.XGrid, k=1)
       
        #test print of KD Tree results
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.scatter(self.XGrid[:, 0], self.XGrid[:, 1], c=distances.flatten())
        plt.colorbar()
        plt.title("Distances of closest point ")
        plt.subplot(122)
        plt.scatter(self.XGrid[:, 0], self.XGrid[:, 1], c=indices.flatten())
        plt.colorbar()
        plt.title("Indices of closest point")
       
        #signed distance function
        for i in (indices):
            self.SDF[i] = (V[i]-indices[i,:]) #*normals[i,:] #(x-p)*n -what is x? normals not needed b/c 2D? 
                
        pass

        

scanner = FakeScanner2D("fish.png")
pos = np.array([100, 100]) # Position of the camera
towards = np.array([1, 1]) # Direction of the camera
fov = np.pi/2 # Field of view of the camera
res = 100 # Resolution of the camera
range_scan = scanner.get_range_scan(pos, towards, fov, res, do_plot=True)
plt.show()


recon = Reconstruction2D(200, 0, 800, 0, 800)
recon.incorporate_scan(pos, towards, fov, range_scan)
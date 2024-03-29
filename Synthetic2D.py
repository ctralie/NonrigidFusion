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
import skimage.measure
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
        range_scan = np.inf*np.ones(res)
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
            V = towards + np.tan(theta)*right
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
        return range_scan, normals


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
        self.XGrid = np.array([X.flatten(), Y.flatten()]).T
       # print(self.XGrid)
        self.SDF = np.nan*np.ones((res, res)) # The signed distance image
        self.weights = np.zeros((res, res))
    
    def get_scan_points(self, pos, towards, fov, res, range_scan, normals):
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
        res: int
            Resolution of the camere
        range_scan: ndarray(N)
            The range scan of the scanner
        normals: ndarray(N, 2)
            The normals at each point of intersection
            (TODO: They are in global coordinates now, should be relative to camera)
        
        Returns
        -------
        V: ndarray(M <= N, 2)
            Points scanned in global coordinates that were actually seen 
            (non-infinite)
        N: ndarray(M <= N, 2)
            Array of corresponding normals
        """
        right = np.array([towards[1], -towards[0]])
        thetas = np.linspace(-fov/2, fov/2, res)
        V = np.ones((res,2)) #will hold vectors
        #recreation of x,y coordinates for each intersection as unit vectors
        for i,theta in enumerate(thetas):
            v = towards + np.tan(theta)*right
            m = np.sqrt(np.sum(v*v))
            v = v / m
           # print(v) #unit vector test
            V[i] = pos + range_scan[i]*v
        V = V[np.isfinite(range_scan), :]
        N = normals[np.isfinite(range_scan), :]
        return V, N

    def incorporate_scan(self, V, N, trunc_dist, do_plot=False):
        """
        Given a range scan, update the signed distance image and the weights to incorporate the new scan.
        Parameters
        ----------
        V: ndarray(M, 2)
            Points scanned in global coordinates that were actually seen 
            (non-infinite)
        N: ndarray(M, 2)
            Array of corresponding normals in global coordinates
        trunc_dist: float
            Threshold at which to truncate
        """
        if V.size == 0:
            return
        tree = KDTree(V)
        distances, indices = tree.query(self.XGrid, k=1)
        indices = indices.flatten()
        distances = np.reshape(distances, (self.res, self.res))

        ## Step 1: Compute the Signed distance function
        # All the points on V that are closest to the corresponding
        # points on XGrid
        P  = V[indices, :]
        N2 = N[indices, :]
        sdf = np.sum((self.XGrid - P)*N2, 1)
        sdf = np.reshape(sdf, (self.res, self.res))
        w = np.zeros_like(sdf)
        w[distances < trunc_dist] = 1

        ## Step 2: Incorporate this signed distance
        ## function into the overall signed distance function
        numerator = np.nanprod(np.array([self.weights, self.SDF]), axis=0)
        numerator = numerator + w*sdf
        self.weights = w + self.weights
        idx = self.weights > 0
        self.SDF[idx] = numerator[idx] / self.weights[idx]
        self.SDF[self.weights == 0] = np.nan

        if do_plot:
            sdf[distances >= trunc_dist] = np.nan
            vmax = np.max(np.abs(sdf[~np.isnan(sdf)]))
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(distances)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.title("Euclidean Distances of Nearest Neighbor")
            plt.subplot(132)
            plt.imshow(sdf, cmap='seismic', vmin=-vmax, vmax=vmax)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.title("Signed distance")
            plt.subplot(133)
            plt.imshow(w)
            plt.gca().invert_yaxis()
            plt.title("Weights")
            plt.colorbar()
            plt.show()
    
    def plot_reconstruction(self, scanner, cameras):
        """
        Plot the reconstruction
        Parameters
        ----------
        scanner: FakeScanner2D
            (Optional) The scanner object that was used to 
            obtain the scans.  If it's provided, the ground
            truth image will be drawn for context
        cameras: list of [{'pos':ndarray(2), 'towards':ndarray(2)}]
            (Optional) If specified a list of cameras that were used to obtain the scans
        """
        sdf = self.SDF
        vmax = np.max(np.abs(sdf[~np.isnan(sdf)]))
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        for camera in cameras:
            pos, towards = camera['pos'], camera['towards']
            if scanner:
                scanner.display_contour()
            plt.scatter([pos[0]], [pos[1]])
            p2 = pos + 20*towards
            plt.plot([pos[0], p2[0]], [pos[1], p2[1]])
        plt.gca().invert_yaxis()
        plt.subplot(132)
        plt.imshow(sdf, cmap='seismic', vmin=-vmax, vmax=vmax)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Signed distance")
        plt.subplot(133)
        plt.imshow(self.weights)
        plt.gca().invert_yaxis()
        plt.title("Weights")
        plt.colorbar()
        plt.show()


def get_arc_test(start_theta, end_theta, res):
    t = np.linspace(start_theta, end_theta, res)
    V = np.zeros((t.size, 2))
    V[:, 0] = 400 + 100*np.cos(t)
    V[:, 1] = 400 + 100*np.sin(t)
    N = np.zeros((t.size, 2))
    N[:, 0] = np.cos(t)
    N[:, 1] = np.sin(t)
    return V, N

tic = time.time()
scanner = FakeScanner2D("fish.png")
recon = Reconstruction2D(200, 0, 800, 0, 800)

fov = np.pi/2 # Field of view of the camera
res = 100 # Resolution of the camera

trunc_dist = 20.0
"""
pos1 = np.array([100, 100]) # Position of the camera
towards1 = np.array([1, 1]) # Direction of the camera
range_scan, normals = scanner.get_range_scan(pos1, towards1, fov, res)
V, N = recon.get_scan_points(pos1, towards1, fov, res, range_scan, normals)
recon.incorporate_scan(V, N, trunc_dist, do_plot=True)

pos2 = np.array([350, 75]) # Position of the camera
towards2 = np.array([0, 1]) # Direction of the camera
range_scan, normals = scanner.get_range_scan(pos2, towards2, fov, res)
V, N = recon.get_scan_points(pos2, towards2, fov, res, range_scan, normals)
recon.incorporate_scan(V, N, trunc_dist, do_plot=True)
"""

fishCenter = np.array([400, 400])
initialPosC = np.array([200,200]) #initial camera position in cartesian coordinates
r = np.sqrt(np.sum((initialPosC-fishCenter)**2)) #radius of circle based off of initial camera position
#initialPosP = np.array([np.sqrt(initialPosC[0]**2 + initialPosC[1]**2), np.arctan(initialPosC[1]/initialPosC[0])])

#thetas = [ 0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, (2*np.pi)/3, (3*np.pi)/4, (5*np.pi)/6, np.pi, (7*np.pi)/6, (5*np.pi)/4, (4*np.pi)/3, (3* np.pi)/2, (5*np.pi)/3, (7*np.pi)/4, (11*np.pi)/6]
#thetas = [np.pi/6, np.pi/4, np.pi/3] #testing a smaller array
num_cameras = 20
thetas = np.linspace(0, 2*np.pi, num_cameras+1)[0:-1]

towards = np.array([1,1])

cameras = [{'pos':initialPosC, 'towards':towards}] #list of [{'pos':ndarray(2), 'towards':ndarray(2)}]
cameras.pop(0)

for theta in thetas:
    towards = np.zeros(2)
    if theta > 0 and theta < np.pi/2:
        towards[0] = -1
        towards [1] = -1
    elif theta >= np.pi/2 and theta < np.pi:
        towards[0] = 1
        towards[1] = -1
    elif theta >= np.pi and theta < (3*np.pi)/2:
        towards[0] = 1
        towards [1] = 1
    else:
        towards[0] = -1
        towards[1] = 1
    
    
    #towards is unit vector of polar coordinates
    #towards[0] = (initialPosP[0] * np.cos(initialPosP[1])) / np.sqrt((initialPosP[0] * np.cos(initialPosP[1]))**2 + initialPosP[0] * np.sin(initialPosP[1])**2)
    #towards[1] = (initialPosP[0] * np.sin(initialPosP[1])) / np.sqrt((initialPosP[0] * np.cos(initialPosP[1]))**2 + initialPosP[0] * np.sin(initialPosP[1])**2)
    
    
    circle_xy = np.array([np.cos(theta), np.sin(theta)])
    xy = r*circle_xy + fishCenter
    towards = -circle_xy
    #towards is unit vector of circle?
    #towards[0] = (r * np.cos(theta)) /  np.sqrt ((r * np.cos(theta))**2 + (r * np.sin(theta))**2)
    #towards [1] = (r * np.sin(theta)) /  np.sqrt ((r * np.cos(theta))**2 + (r * np.sin(theta))**2)
   
   
    #xy = np.array([initialPosP[0]*np.cos(initialPosP[1]), initialPosP[0]*np.sin(initialPosP[1])])
    cameras.append({'pos':xy, 'towards':towards})
    
    range_scan, normals = scanner.get_range_scan(xy, towards, fov, res)
    V, N = recon.get_scan_points(xy, towards, fov, res, range_scan, normals)
    recon.incorporate_scan(V,N, trunc_dist, do_plot=False)
    
   

recon.plot_reconstruction(scanner, cameras)

toc = time.time()
print("Elapsed Time: %.3g"%(toc-tic))

plt.figure(figsize=(8, 8))
contours = skimage.measure.find_contours(recon.SDF, 0)
for c in contours:
    plt.plot(c[:, 1], c[:, 0])
plt.axis('equal')
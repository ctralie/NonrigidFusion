"""
Purpose: To simulate a 2D range camera by finding the largest
contour on an image and using it as the object of interest
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import skimage
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
        max_dist = np.sqrt(W**2 + H**2)
        ring = LinearRing(self.contour)
        line = LineString([pos, pos+towards*max_dist])
        x = line.intersection(ring)
        if do_plot:
            self.display_contour()
            plt.scatter([pos[0]], [pos[1]])
            if len(x) > 0:
                # If there was an intersection, draw the closest one
                x = np.array(x[0]) # Extract closest intersection
                segment = np.array([pos, x])
                plt.plot(segment[:, 0], segment[:, 1])
                plt.scatter(x[0], x[1])
            



scanner = FakeScanner2D("fish.png")
pos = np.array([100, 100])
towards = np.array([1, 1])
fov = np.pi/2
res = 100
scanner.get_range_scan(pos, towards, fov, res, do_plot=True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg 
from sklearn.neighbors import KDTree
import skimage
from skimage import measure
import json
import time

class FakeScanner3D(object):
    def __init__(self, filename):
        """
        Load in a .json file generated from ggslac which
        contains all of the depth images, normal images, and
        camera information
        Parameters
        ----------
        filename: string
            Path to JSON scan file
        near: float
            Near dist to filter out the boundary between
            the object and the background due to antialiasing
        """
        print("Loading...")
        data= json.load(open(filename, "r"))
        print("Finished loading")
        self.W = data['width']
        self.H = data['height']
        self.fovx = data['fovx']
        self.fovy = data['fovy']
        self.far = data['far']
        self.allNormals = data['allNormals']
        self.allDepth = data['allDepth']
        self.cameras = data['cameras']
        self.num_scans = len(self.allNormals)
        self.unpack_scans()
    
    def unpack_scans(self):
        """
        Reshape the depth and normal scans and unpack the 
        information from the [0, 255] RGB channels
        """
        W = self.W
        H = self.H
        allNormals = []
        allDepth = []
        for i in range(self.num_scans):
            # Extract normal scan
            normals = self.allNormals[i]
            normals = np.reshape(normals, (H, W, 4))
            normals = np.array(normals[:, :, 0:3], dtype=float)/255
            normals = normals*2 - 1
            # Just in case they aren't normalized
            mags = np.sqrt(np.sum(normals**2, 2)) 
            normals = normals/mags[:, :, None]

            # Extract depth scan
            depth = self.allDepth[i]
            depth = np.reshape(depth, (H, W, 4))
            depth = np.array(depth[:, :, 0:2], dtype=float)/255
            depth = depth[:, :, 0] + depth[:, :, 1]/256
            depth = depth*(256*256)/(256*256-1)
            depth[depth == 0] = np.nan

            allNormals.append(normals)
            allDepth.append(depth)
            for key in self.cameras[i].keys():
                self.cameras[i][key] = np.array(self.cameras[i][key])
            self.cameras[i]['fovx'] = self.fovx
            self.cameras[i]['fovy'] = self.fovy
        self.allNormals = allNormals
        self.allDepth = allDepth

    def get_scan(self, i):
        """
        Get a particular scan in the list of scans
        Parameters
        ----------
        i: int
            Index of the scan
        Returns
        -------
        {
            'depth': ndarray(M, N)
                The depth scan,
            'normals': ndarray(M, N, 3)
                The normals for each depth point,
            'camera': {
                'pos': ndarray(3)
                    Position of the camera,
                'up': ndarray(3)
                    Up vector of the camera,
                'right': ndarray(3)
                    Right vector of the camera,
                'fovx': float
                    Field of view in x,
                'fovy': float
                    Field of view in y
            }
        }
        """
        if i >= self.num_scans:
            raise Exception("Requested scan beyond range")
        return {'depth':self.allDepth[i], 'normals':self.allNormals[i], 'camera':self.cameras[i]}
    
    def plot_scans(self):
        """
        Make a video of all of the scans
        """
        vmin = np.inf
        vmax = 0
        for depth in self.allDepth:
            vmin = min(vmin, np.nanmin(depth))
            vmax = max(vmax, np.nanmax(depth))
        print("vmin = %.3g"%vmin)
        print("vmax = %.3g"%vmax)
        fac = 1
        plt.figure(figsize=(fac*14, fac*5))
        for i, (depth, normals) in enumerate(zip(self.allDepth, self.allNormals)):
            plt.clf()
            plt.subplot(121)
            plt.imshow(depth, vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.title("Depth Scan {}".format(i))
            plt.subplot(122)
            plt.imshow((normals+1)/2)
            plt.title("Normals {}".format(i))
            plt.savefig("{}.png".format(i), bbox_inches='tight')

def make_volume_video(X, prefix):
    """
    Make a video of slices of a volume sweeping over
    the volume
    Parameters
    ----------
    X: ndarray(N1, N2, N3)
        A volumetric array, possibly with nans
    prefix: string
        The prefix of each png file
    """
    vmin = np.nanmin(X)
    vmax = np.nanmax(X)
    for i in range(X.shape[0]):
        plt.clf()
        plt.imshow(X[i, :, :], vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.savefig("{}.png".format(i), bbox_inches='tight')


class Reconstruction3D(object):
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
    zmin: float
        Min z coordinate
    zmax: float
        Max z coordinate
    XGrid: ndarray(res*res*res, 3)
        The locations of the points on the grid
    SDF: ndarray(res, res, res)
        The signed distance image
    weights: ndarray(res, res, res)
        Weights for the SDF points (see more info in Curless/Levoy)
    """
    
    def __init__(self, res, xmin, xmax, ymin, ymax, zmin, zmax):
        # Store grid information as local variables
        self.res = res
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        pixx = np.linspace(xmin, xmax, res)
        pixy = np.linspace(ymin, ymax, res)
        pixz = np.linspace(zmin, zmax, res)
        X, Y, Z = np.meshgrid(pixx, pixy, pixz)
        self.XGrid = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        self.SDF = np.nan*np.ones((res, res, res)) # The signed distance image
        self.weights = np.zeros((res, res, res))
    
    def get_scan_points(self, depth, normals, camera):
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
            The normals at each point of intersection, 
            *relative to the camera*
        
        Returns
        -------
        V: ndarray(M <= N, 2)
            Points scanned in global coordinates that were actually seen 
            (non-infinite)
        N: ndarray(M <= N, 2)
            Array of corresponding normals
        """
        pass

    def incorporate_scan(self, V, N, trunc_dist, do_plot=False):
        """
        Given a range scan, update the signed distance image and the weights to incorporate the new scan.
        Parameters
        ----------
        V: ndarray(M, 3)
            Points scanned in global coordinates that were actually seen 
            (non-infinite)
        N: ndarray(M, 3)
            Array of corresponding normals in global coordinates
        trunc_dist: float
            Threshold at which to truncate
        """
        if V.size == 0:
            return
        tree = KDTree(V)
        distances, indices = tree.query(self.XGrid, k=1)
        indices = indices.flatten()
        distances = np.reshape(distances, (self.res, self.res, self.res))

        ## Step 1: Compute the Signed distance function
        # All the points on V that are closest to the corresponding
        # points on XGrid
        P  = V[indices, :]
        N2 = N[indices, :]
        sdf = np.sum((self.XGrid - P)*N2, 1)
        sdf = np.reshape(sdf, (self.res, self.res, self.res))
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


scanner = FakeScanner3D("Data/homer.json")
scanner.plot_scans()
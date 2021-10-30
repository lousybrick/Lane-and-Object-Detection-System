import numpy as np 
import cv2

# Class to store camera calibration data
class calibration():
    def __init__(self, obj_pts, img_pts, shape):
        self.ret, self.M, self.dist, self.rvecs, self.tvecs =\
        cv2.calibrateCamera(obj_pts, img_pts, shape, None, None)

# Class to store perspective transform matrices
class transform():
    def __init__(self, src, dst):       
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        
# Class to store lane detection data
class lane():
    def __init__(self):
        self.detected = False
        self.x = None
        self.y = None
        self.fit = None
        
# Class to store color and gradient feature extraction parameters
class parameters():
    def __init__(self):
        self.spat_size = (32, 32)
        self.hist_bins = 32
        self.orient = 8
        self.pxs_cell = (8, 8)
        self.cells_block = (2, 2)

# Class to store vehicle detection data
class vehicle():
    def __init__(self):
        self.scaler = None
        self.clf = None
        self.heatmaps = None
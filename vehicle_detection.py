import numpy as np 
import cv2 
import matplotlib.image as mpimg
from skimage.feature import hog
import classes

# Initializing classes
src = np.float32([[685, 450], [1090, 720], [190, 720], [595, 450]])
dst = np.float32([[990, 0], [990, 720], [290, 720], [290, 0]])
trans = classes.transform(src, dst)

left = classes.lane()
right = classes.lane()

par = classes.parameters()
car = classes.vehicle()

# Compute spatial binning feature vector
def get_spat_feat(img):
    return cv2.resize(img, dsize=par.spat_size).ravel() 

# Compute color histogram feature vector
def get_hist_feat(img):
    hist_feat = []
    # Compute histogram for each color channel
    for channel in np.arange(img.shape[2]):
        hist_feat.append(np.histogram(img[:,:,channel], bins=par.hist_bins)[0])
    return np.concatenate(hist_feat)

# Compute HOG feature vector
def get_hog_feat(img, feat_vec):
    hog_feat = []
    # Compute HOG for each color channel
    for channel in np.arange(img.shape[2]):
        hog_feat.append(hog(img[:,:,channel], orientations=par.orient, pixels_per_cell=par.pxs_cell,
                            cells_per_block=par.cells_block, visualise=False, transform_sqrt=True,
                            feature_vector=feat_vec))
    if feat_vec == True:
        return np.concatenate(hog_feat)
    else: return hog_feat

# Extract feature vectors from images
def extract_feat(fnames):
    features = []
    for file in fnames:
        img = mpimg.imread(file)
        # Convert image to YCrCb color space
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        spat_feat = get_spat_feat(img)
        hist_feat = get_hist_feat(img)
        hog_feat = get_hog_feat(img, feat_vec=True)
        # Combine feature vectors
        features.append(np.concatenate((spat_feat, hist_feat, hog_feat)))
    return features

# Find car bounding boxes
def find_cars(img, ystart, ystop, scale):
    # Crop image
    img = img[ystart:ystop,:,:]
    
    # Convert image to YCrCb color space
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    
    # Scale image
    if scale != 1:
        img = cv2.resize(img, (np.int(img.shape[1]/scale), np.int(img.shape[0]/scale)))
    
    # Compute number of HOG cells in x and y directions
    xcells = (img.shape[1]//par.pxs_cell[0])
    ycells = (img.shape[0]//par.pxs_cell[0])
    
    # Training dataset image size
    window = 64
    
    # Compute number of HOG cells and blocks in window
    cells_window = (window//par.pxs_cell[0])
    block_steps = cells_window - par.cells_block[0] + 1
    
    # Compute number of steps in x and y directions
    cells_step = 1
    xsteps = (xcells - cells_window)//cells_step
    ysteps = (ycells - cells_window)//cells_step
    
    hog = get_hog_feat(img, feat_vec = False)
    
    bbox_list = []
    for xb in np.arange(xsteps):
        for yb in np.arange(ysteps):
            # Search window top left coordinate points in cell units
            ypos = yb*cells_step
            xpos = xb*cells_step
            
            # Within search window extract HOG subsample feature vector for each color channel
            hog_feat = []
            for channel in np.arange(img.shape[2]):
                hog_feat.append(hog[channel][ypos:ypos+block_steps, xpos:xpos+block_steps])
            hog_feat = np.concatenate(hog_feat).ravel()
    
            # Search window top left coordinate points in pixel units
            xleft = xpos*par.pxs_cell[0]
            ytop = ypos*par.pxs_cell[0]
            
            # Create search window
            sub_img = cv2.resize(img[ytop:ytop+window, xleft:xleft+window], (window, window))
            
            spat_feat = get_spat_feat(sub_img)
            hist_feat = get_hist_feat(sub_img)
            
            # Combine and scale feature vectors
            features = np.concatenate((spat_feat, hist_feat, hog_feat))
            scaled_features = car.scaler.transform(features)
            
            # Predict class and save confidence
            confidence = car.clf.decision_function(scaled_features)
            
            # Save search window coordinate points
            if confidence > 0.5:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))) 
    return bbox_list

# Create heatmap from bounding boxes
def create_heatmap(heatmap, bbox_list):
    for bbox in bbox_list:
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
    return heatmap
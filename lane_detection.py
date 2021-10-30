import numpy as np 
import cv2
import classes

# Initializing classes
src = np.float32([[685, 450], [1090, 720], [190, 720], [595, 450]])
dst = np.float32([[990, 0], [990, 720], [290, 720], [290, 0]])
trans = classes.transform(src, dst)

left = classes.lane()
right = classes.lane()

par = classes.parameters()
car = classes.vehicle()

# Thresholding function
def threshold(img, thresh):
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    
    return binary

# Thresholded absolute Sobel gradient
def grad_thresh(img, orient, ksize, thresh):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img[:, :, 0], cv2.CV_64F, 1, 0, ksize))
    
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img[:, :, 0], cv2.CV_64F, 0, 1, ksize))
    
    scaled_sobel = np.uint8(abs_sobel*(255/np.max(abs_sobel)))
    
    return threshold(scaled_sobel, thresh)

# Thresholded magnitude of Sobel gradient
def mag_thresh(img, ksize, thresh):
    sobelx = cv2.Sobel(img[:, :, 0], cv2.CV_64F, 1, 0, ksize)
    sobely = cv2.Sobel(img[:, :, 0], cv2.CV_64F, 0, 1, ksize)
    
    mag_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    
    scaled_sobel = np.uint8(mag_sobelxy*(255/np.max(mag_sobelxy)))

    return threshold(scaled_sobel, thresh)

# Thresholded direction of Sobel gradient
def dir_thresh(img, ksize, thresh):
    abs_sobelx = np.absolute(cv2.Sobel(img[:, :, 0], cv2.CV_64F, 1, 0, ksize))
    abs_sobely = np.absolute(cv2.Sobel(img[:, :, 0], cv2.CV_64F, 0, 1, ksize))
    
    dir_sobelxy = np.arctan2(abs_sobely, abs_sobelx)
    
    return threshold(dir_sobelxy, thresh)

# Warp image perspective
def warp(img, view):
    # Warp image perspective into bird's eye view
    if view == 'b':
        warped = cv2.warpPerspective(img, trans.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        
    # Warp image perspective into driver's view
    elif view == 'd':
        warped = cv2.warpPerspective(img, trans.Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
    return warped

# Find initial lane locations
def find_initial(img):
    histogram = np.sum(img[int(img.shape[0]*(3/4)):, :], axis=0) 
    midpoint = histogram.shape[0]//2
    
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint 

    left.x, left.y = find_initial_sub(img, leftx_base)
    right.x, right.y = find_initial_sub(img, rightx_base)
    
    left.fit = np.polyfit(left.y, left.x, 2)
    right.fit = np.polyfit(right.y, right.x, 2)
    
    left.detected = True
    right.detected = True
    
    return None

# Find initial lane locations - sub function
def find_initial_sub(img, xbase):
    num_windows = 9   
    window_height = img.shape[0]//num_windows # 80
    margin = 80
    min_pix = 800
    
    non_zeroy = np.array(img.nonzero()[0])
    non_zerox = np.array(img.nonzero()[1])
    
    lane_inds = []
    shift = 0
    
    # Sliding window technique
    for window in range(num_windows):
        winy_low = img.shape[0] - (window + 1)*window_height # Window lower bound
        winy_high = img.shape[0] - window*window_height # Window upper bound

        winx_left = xbase - margin # Window left bound
        winx_right = xbase + margin # Window right bound

        lane_inds_cur = ((non_zeroy >= winy_low) & (non_zeroy < winy_high) &\
                          (non_zerox >= winx_left) & (non_zerox < winx_right)).nonzero()[0]

        lane_inds.append(lane_inds_cur)
        
        # Center window on average of detected non-zero pixels
        if len(lane_inds_cur) > min_pix:
            xbase_new = np.int(np.mean(non_zerox[lane_inds_cur]))
            shift = xbase_new - xbase
            xbase = xbase_new
        
        # Shift window by previous amount
        else:
            xbase += shift
        

    lane_inds = np.concatenate(lane_inds)

    return non_zerox[lane_inds], non_zeroy[lane_inds]

# Find lane locations using margin around previous lane locations
def find_next(img):
    leftx, lefty = find_next_sub(img, left.fit)
    rightx, righty = find_next_sub(img, right.fit)
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Check whether current lanes are within margin of previous lanes
    if np.all([((left_fit[2] > left.fit[2]*0.8) & (left_fit[2] < left.fit[2]*1.2)),\
               ((right_fit[2] > right.fit[2]*0.8) & (right_fit[2] < right.fit[2]*1.2))]):
        
        # Compute weighted average of current and previous polynomial coefficients
        # based on number of non-zero pixels detected
        left_weights = np.array([(len(leftx)/(len(leftx) + len(left.x))),\
                                 (len(left.x)/(len(leftx) + len(left.x)))])

        right_weights = np.array([(len(rightx)/(len(rightx) + len(right.x))),\
                                 (len(right.x)/(len(rightx) + len(right.x)))])

        left.fit = np.average((left_fit, left.fit), axis = 0, weights = left_weights)
        right.fit = np.average((right_fit, right.fit), axis = 0, weights = right_weights)

        left.x, left.y = leftx, lefty
        right.x, right.y = rightx, righty
    
    # If current lanes outside of margin of previous lanes detect lanes from scratch
    else:
        find_initial(img)
    
    return None

def find_next_sub(img, fit):
    margin = 80
    
    non_zeroy = np.array(img.nonzero()[0])
    non_zerox = np.array(img.nonzero()[1])
    
    # Detect non-zero pixels within previously computed polynomial +/- margin
    lane_inds = ((non_zerox > (fit[0]*(non_zeroy**2) + fit[1]*non_zeroy + fit[2] - margin)) &\
                 (non_zerox < (fit[0]*(non_zeroy**2) + fit[1]*non_zeroy + fit[2] + margin)))
    
    return non_zerox[lane_inds], non_zeroy[lane_inds]

# Calculate road curvature radius and vehicle relation to lane center
def info(shape):
    
    # Lane base x-points
    leftx_base = np.mean(left.x[(left.y >= shape[0]*(3/4))])
    rightx_base = np.mean(right.x[(right.y >= shape[0]*(3/4))]) 
    
    # Pixel to meter conversion factors
    xconv = 3.7/(rightx_base - leftx_base)
    yconv = 30/720
    
    # Lane polynomial coefficients
    left_fit = np.polyfit(left.y*yconv, left.x*xconv, 2)
    right_fit = np.polyfit(right.y*yconv, right.x*xconv, 2)
    
    # Average of lane polynomial coefficients
    fit = np.mean((left_fit, right_fit), axis = 0)
    
    # Road curvature radius
    curve_rad = (((1 + (2*fit[0]*shape[0]*yconv + fit[1])**2)**1.5)/np.absolute(2*fit[0]))
    
    # Vehicle relation to lane center
    off_center = (shape[1]/2 - np.mean((leftx_base, rightx_base)))*xconv
    
    return curve_rad, off_center
import numpy as np 
import cv2
import classes
import lane_detection

# Initializing classes
src = np.float32([[685, 450], [1090, 720], [190, 720], [595, 450]])
dst = np.float32([[990, 0], [990, 720], [290, 720], [290, 0]])
trans = classes.transform(src, dst)

left = classes.lane()
right = classes.lane()

par = classes.parameters()
car = classes.vehicle()

# Draw identified lane and road information on image
def draw_lane(img):
    # Create blank canvas to draw lane on
    blank = np.zeros_like(img).astype(np.uint8)
    
    # Create y-points
    ploty = np.linspace(0, blank.shape[0] - 1, blank.shape[0])
    
    # Compute x-points
    leftx = left.fit[0]*ploty**2 + left.fit[1]*ploty + left.fit[2]
    rightx = right.fit[0]*ploty**2 + right.fit[1]*ploty + right.fit[2]
    
    # Combine lane points
    left_pts = np.array([np.transpose(np.vstack([leftx, ploty]))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((left_pts, right_pts))
    
    # Draw lane
    lane = cv2.fillPoly(blank, np.int_([pts]), (0, 255, 0))
    
    # Project lane onto road
    projected = lane_detection.warp(lane, view = 'd')
    
    # Combine image with projected lane
    return cv2.addWeighted(img, 1, projected, 0.3, 0)
    
# Draw road curvature radius and vehicle relation to lane center on image
def draw_info(img):
    curve_rad, off_center = lane_detection.info(img.shape)
    
    curve_text = 'Curvature radius = {:d}m'.format(int(curve_rad))
    
    if off_center < 0:
        off_text = 'Vehicle {:.2f}m left of center'.format(abs(off_center))
    
    elif off_center > 0:
        off_text = 'Vehicle {:.2f}m right of center'.format(abs(off_center))

    # Draw curvature radius on image
    cv2.putText(img, curve_text, org = (10, 30), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1, thickness = 2, color = (0, 255, 0), bottomLeftOrigin = False)
    
    # Draw vehicle relation to lane center on image
    return cv2.putText(img, off_text, org = (10, 60), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale = 1, thickness = 2, color = (0, 255, 0), bottomLeftOrigin = False)

def draw_bbox(img, cars):
    # Draw bounding box for each car
    for car in np.arange(1, cars[1] + 1):
        non_zerox = (cars[0] == car).nonzero()[1]
        non_zeroy = (cars[0] == car).nonzero()[0]
        bbox = (np.min(non_zerox), np.min(non_zeroy)), (np.max(non_zerox), np.max(non_zeroy))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,256), 4)
        
    return img
import glob
import time
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import classes
import lane_detection
import vehicle_detection
import draw

# Initializing classes
src = np.float32([[685, 450], [1090, 720], [190, 720], [595, 450]])
dst = np.float32([[990, 0], [990, 720], [290, 720], [290, 0]])
trans = classes.transform(src, dst)

left = classes.lane()
right = classes.lane()

par = classes.parameters()
car = classes.vehicle()

# Calibrate camera
# Load calibration image filenames
calibration_fnames = glob.glob('camera_cal/calibration*.jpg')

numx = 9 # Number of inner row corners
numy = 6 # Number of inner column corners

cal_obj_pts = []
cal_img_pts = []

# Create array of known object points
obj_pts = np.zeros((numx*numy, 3), np.float32)
obj_pts[:, :2] = np.mgrid[0:numx, 0:numy].T.reshape(-1, 2)

# Loop over calibration images
for fname in calibration_fnames:
    img = mpimg.imread(fname) # Load image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert image to grayscale
    ret, img_pts = cv2.findChessboardCorners(gray, (numx, numy), None) # Detect image points
        
    # If chessboard corners detected, append object and image points to those previously detected
    if ret == True: 
        cal_obj_pts.append(obj_pts)
        cal_img_pts.append(img_pts)

# Create an instance of the camera calibration data class
cal = classes.calibration(cal_obj_pts, cal_img_pts, img.shape[0:2])


# Extract Features
# Load car and non-car image filenames
car_fnames = glob.glob('training_data/vehicles/vehicle*.jpg')
non_car_fnames = glob.glob('training_data/non_vehicles/non_vehicle*.jpg')

print('Extracting features...')

# Extract color and gradient features from car and non-car images
car_feat = vehicle_detection.extract_feat(car_fnames)
non_car_feat = vehicle_detection.extract_feat(non_car_fnames)

# Combine car and non-car features
X = np.vstack((car_feat, non_car_feat)).astype(np.float64)

print('Standardizing features...')

# Standardize features by zero mean centering and scaling to standard deviation
car.scaler = StandardScaler().fit(X)
scaled_X = car.scaler.transform(X)

# Create labels for car and non-car features
y = np.hstack((np.ones(len(car_feat), dtype = np.int8), np.zeros(len(non_car_feat), dtype = np.int8)))

print('Creating training and test subsets...')

# Randomly split features and labels into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=np.random.randint(0, 100))

print('Number of training examples = ', len(X_train))
print('Number of test examples = ', len(X_test))


# Train Classifier
# Initiate Linear SVC classifier
car.clf = LinearSVC()

print('Training...')

start = time.time()
car.clf.fit(X_train, y_train)
stop = time.time()

print('Training Time = {:.3f} s'.format(stop - start))
print('Training Accuracy = {:.3f}'.format(car.clf.score(X_train, y_train)))
print('Test Accuracy = {:.3f}'.format(car.clf.score(X_test, y_test)))

# Lane Detection Pipeline
def lane_pipeline(img):    
    # Threshold image
    binary_gradx = lane_detection.grad_thresh(img, orient = 'x', ksize = 3, thresh = (20, 255))
    binary_grady = lane_detection.grad_thresh(img, orient = 'y', ksize = 3, thresh = (20, 255))
    
    binary_grad_mag = lane_detection.mag_thresh(img, ksize = 3, thresh = (40, 255))
    binary_grad_dir = lane_detection.dir_thresh(img, ksize = 3, thresh = (0.7, 1.3))
    
    binary_rch = lane_detection.threshold(img[:, :, 0], thresh = (140, 255))
    binary_sch = lane_detection.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 2],\
                           thresh = (100, 255))
    
    # Combine binary images
    combined = np.zeros_like(binary_gradx)
    combined[((binary_gradx == 1) & (binary_grady == 1)) |\
                ((binary_grad_mag == 1) & (binary_grad_dir == 1)) |\
                ((binary_rch == 1) & (binary_sch == 1))] = 1
    
    # Warp image perspective into bird's eye view
    warped = lane_detection.warp(combined, view = 'b')
    
    # Find lane locations from scratch
    if left.detected == False & right.detected == False:
        lane_detection.find_initial(warped)
    
    # Find lane locations using margin around previously found lane locations
    elif left.detected == True & right.detected == True:
        lane_detection.find_next(warped)
   
    return None

# Vehicle Detection Pipeline
def vehicle_pipeline(img):
    # Parameters
    ystart_list = [400, 400]
    ystop_list = [480, 528]
    scale_list = [1.25, 1.5]
    thresh = 4
    frames = 4
    
    # Find car bounding boxes
    bbox_list = [] 
    for ystart, ystop, scale in zip(ystart_list, ystop_list, scale_list):
        bbox = vehicle_detection.find_cars(img, ystart, ystop, scale)
        if len(bbox) > 0:
            bbox_list.append(bbox)

    # Create a heatmap if bounding boxes were found
    if len(bbox_list) > 0:
        bbox_list = np.concatenate(bbox_list)
        car.heatmaps.append(vehicle_detection.create_heatmap(np.zeros_like(img[:,:,0]), bbox_list))
    
    # Combine last n stored heatmaps
    heatmap = np.zeros_like(img[:,:,0])
    for i in range(len(car.heatmaps)):
        heatmap += car.heatmaps[i]
    
    # Delete first in n stored heatmaps
    if len(car.heatmaps) > frames:
        del car.heatmaps[0]
    
    # Threshold heatmap
    heatmap[heatmap < thresh] = 0

    # Label individual heatmap regions
    return label(heatmap)


# Combined Pipeline 
def pipeline(img):
    
    # Undistort image using previously computed camera calibration matrix
    img = cv2.undistort(img, cal.M, cal.dist, None, cal.M)
    
    # Detect lane
    lane_pipeline(img)
    
    # Detect vehicles
    cars = vehicle_pipeline(img)
    
    # Draw lane, info, and bounding boxes
    lane = draw.draw_lane(img)
    info = draw.draw_info(lane)    
    return draw.draw_bbox(info, cars)

def main():
    # Output 
    car.heatmaps = []
    video_output = 'output/output_video.mp4'
    video_input = VideoFileClip('input/input_video.mp4')
    video_clip = video_input.fl_image(pipeline)
    video_clip.write_videofile(video_output, audio=False)

if __name__=="__main__":
    main()
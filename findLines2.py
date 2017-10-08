# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 04:02:21 2017

@author: Toshiharu
"""

class Line():
    def __init__(self):
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations, only used in guided search
        self.best_fit = np.array([])  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([]) 
        #radius of curvature of the line in some units
        self.radius = None 
        #distance in meters of vehicle center from the line
        self.base = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.fitPoints = None  
        #Convertion form pixels to meters (x dir)
        self.x_factor = 3.7/480
        #Convertion form pixels to meters (y dir)
        self.y_factor = 3/100
        # bottom of the figure
        self.y_limit = 719
        # points to fit
        self.points = None
        self.yeval = 720/2
        
        self.detected = False  # Used to select between the full of guided seach
        self.timesMissed = 0  # Used only on the GuidedSearch function
    
#    def fitLine(self):
#        return  np.polyfit(self.fitPoints[:,1] , self.fitPoints[:,0], 2)
        
    def linePlot(self):
        ploty = np.linspace(0,self.y_limit-1, self.y_limit )
        plotx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        return plotx,ploty
    
   
#def findFull(warped_binary, nwindows = 9, windowWidth=100, minpix=50,draw = False):

import cv2

from imageProcessing import perspectiveCal,Calibration,birdEye
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from imageFilter import Multifilter

# Load camera lens distortion correction parameters
import pickle

calibration_par = 'camera_cal/calibration.p'
dist_pickle = pickle.load( open( calibration_par, "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Load perpective matrices
perspective_mat = 'camera_cal/perspective.p'
dist_pickle = pickle.load( open( perspective_mat, "rb" ) )
M = dist_pickle["M"]
Minv = dist_pickle["Minv"]
########################################################################################
####   LOAD IMAGE
#########################################################################################
fname = 'test_images/test8.jpg'
img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
top_down = birdEye(img, mtx, dist,M)
binary_warped = Multifilter(top_down,s_thresh=(210, 255),b_thresh=(155,200),l_thresh=(220,255),sxy_thresh=(30,100), draw=False)

###########################################################################################
nwindows = 9
windowWidth=100
minpix=50
###########################################################################################
# HISTOGRAM : search in the quarter bottom first
###########################################################################################
histogram = np.sum(binary_warped[int(3*binary_warped.shape[0]/4):,:], axis=0)
xHist = np.linspace(1,1280,1280,dtype=int)
plt.plot(xHist,histogram)

# Create an output image to draw on and  visualize the result
#out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

###########################################################################################
# FIND BASE POINTS These will be the starting point for the left and right lines
###########################################################################################

midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
# show base points in histogram as red lines
plt.axvline(x=leftx_base,color='red')
plt.axvline(x=rightx_base,color='red')
plt.show()

# Set height of windows
windowHeight = np.int(binary_warped.shape[0]/nwindows)
print(windowHeight)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base

# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

#######################################################################################
### BATCH TESTING
#######################################################################################
import glob

images = []
#images = glob.glob('test_images/straight_lines*.jpg')
images = glob.glob('test_images/test*.jpg')

for idx, fname in enumerate(images):
    print(fname)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    top_down = birdEye(img, mtx, dist,M)
    binary_warped = Multifilter(top_down,s_thresh=(210, 255),b_thresh=(155,200),l_thresh=(220,255),sxy_thresh=(30,100), draw=False)
    
    leftLine = Line()
    rightLine = Line()

    # HISTOGRAM
    histogram = np.sum(binary_warped[int(3*binary_warped.shape[0]/4):,:], axis=0)
    
#    histogram = np.sum(binary_warped, axis=0)
    
    xHist = np.linspace(1,1280,1280,dtype=int)
    
    # Find where the seach will start
    flag = True
    midOffset = 0
    midpoint = np.int(histogram.shape[0]/2)
    
    leftLine.base = np.argmax(histogram[:midpoint])
    rightLine.base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Reset initial position for each window
    leftx_current = leftLine.base
    rightx_current = rightLine.base
#######################################################################################
    leftCentroids=np.zeros((nwindows,2))   # Stores the centroids for each iterated window
    rightCentroids=np.zeros((nwindows,2))
    
    leftDetectedWindows = 0
    rightDetectedWindows = 0
    
    debug = True
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    for window in range(nwindows):
    #window=0
        # Left Window
        leftBottom = int(binary_warped.shape[0]-window*windowHeight) 
        leftTop = int(binary_warped.shape[0]-(window+1)*windowHeight)
        leftLeft = int(leftx_current-windowWidth/2)
        leftRight = int(leftx_current+windowWidth/2)
        
        leftWindow = binary_warped[leftTop:leftBottom,leftLeft:leftRight]
        leftNPoints = np.sum(leftWindow)

        # Right Window
        rightBottom = int(binary_warped.shape[0]-window*windowHeight )
        rightTop = int(binary_warped.shape[0]-(window+1)*windowHeight)
        rightLeft = int(rightx_current-windowWidth/2)
        rightRight = int(rightx_current+windowWidth/2)
        
        rightWindow = binary_warped[rightTop:rightBottom,rightLeft:rightRight]
        rightNPoints = np.sum(rightWindow)
        
        if(leftNPoints>minpix):
            leftx_current = np.int(np.mean(np.nonzero(leftWindow)[1])) + leftLeft
            leftDetectedWindows += 1    # Update number of segments in left line
        
        if(rightNPoints>minpix):
            rightx_current = np.int(np.mean(np.nonzero(rightWindow)[1])) + rightLeft
            rightDetectedWindows += 1    # Update number of segments in left line
            
        # Store centroids of each window
        currentHeight = (leftTop+leftBottom)/2
        
        leftCentroids[window][0] = leftx_current 
        leftCentroids[window][1] = currentHeight
        
        rightCentroids[window][0] = rightx_current 
        rightCentroids[window][1] = currentHeight

   #    print(leftBottom, leftTop, leftLeft, leftRight)
############################################################################
### DEBUG
############################################################################
        
        if (debug== True):
            cv2.rectangle(out_img,(leftLeft,leftTop),(leftRight,leftBottom-1),
                (0,255,0), 4) 
            
            cv2.rectangle(out_img,(rightLeft,rightTop),(rightRight,rightBottom-1),
                (0,255,0), 4)
            
            print('Points in left window: ',window,leftNPoints)
            print('Points in right window: ',window,rightNPoints)


############################################################################
### DEBUG END
############################################################################ 
     
        # Fit a second order polynomial to each
    leftLine.current_fit  = np.polyfit(leftCentroids[:,1] , leftCentroids[:,0], 2)
    rightLine.current_fit = np.polyfit(rightCentroids[:,1], rightCentroids[:,0], 2)
    
    #Find the Line parameters in meters
    leftLineFit_world   = np.polyfit(leftCentroids[:,1]*leftLine.y_factor , leftCentroids[:,0]*leftLine.x_factor, 2)
    rightLineFit_world  = np.polyfit(rightCentroids[:,1]*rightLine.y_factor, rightCentroids[:,0]*rightLine.x_factor, 2)   

    if(leftDetectedWindows>4 ):
        leftLine.detected = True

    if( rightDetectedWindows>4 ):
        rightLine.detected = True
        
    ##########################################        
    if (debug== True):
        leftLine.best_fit = leftLine.current_fit
        rightLine.best_fit = rightLine.current_fit
        
        left_fitx,ploty = leftLine.linePlot()
        right_fitx,ploty = rightLine.linePlot()
        
        plt.plot(left_fitx, ploty, color='red')
        plt.plot(right_fitx, ploty, color='red')
        
        plt.imshow(out_img)
        plt.show()  

    print(leftLine.detected, rightLine.detected)    
    
    
#def findGuided():
    

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 04:02:21 2017

@author: Toshiharu
"""

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([])  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([]) 
        #radius of curvature of the line in some units
        #self.radius = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.fitPoints = None  
        #Convertion form pixels to meters (x dir)
        self.x_factor = 0
        #Convertion form pixels to meters (y dir)
        self.y_factor = 0
        # bottom of the figure
        self.y_limit = 719
        # points to fit
        self.points = None
        self.yeval = 720/2
        self.miss = 5  # max number of miss before calling findFull again
    
#    def fitLine(self):
#        return  np.polyfit(self.fitPoints[:,1] , self.fitPoints[:,0], 2)
        
    def linePlot(self):
        ploty = np.linspace(0,self.y_limit-1, self.y_limit )
        plotx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        return plotx,ploty
    
def gaussianBlur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

#        rho = 1 # distance resolution in pixels of the Hough grid
#        theta = np.pi/180 # angular resolution in radians of the Hough grid
#        threshold =30  #1   # minimum number of votes (intersections in Hough grid cell)
#        min_line_length = 20 #5 #minimum number of pixels making up a line
#        max_line_gap = 15 #1   # maximum gap in pixels between connectable line segments
#        line_image = np.copy(image)*0 # creating a blank to draw lines on

def hough_lines(img, rho=5, theta=np.pi/180, threshold=30, min_line_len=50, max_line_gap=5):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
#    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img = np.copy(img)*0
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), (255,0, 0), thickness=2)
## For debugging   
#    comb_img = cv2.addWeighted(img, 0.8, line_img, 1,0.5)
    plt.figure()        
    plt.imshow(line_img)
#   draw_lines(line_img, lines)
    return lines

    
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
#test_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
test_img = gaussianBlur(binary_warped,kernel_size=5)
plt.imshow(test_img)
plt.show()
lines = hough_lines(test_img,rho=5, theta=np.pi/180, threshold=40, min_line_len=30, max_line_gap=5)
x1,y1,x2,y2 = lines[:,:,0], lines[:,:,1], lines[:,:,2], lines[:,:,3]
xPoints = np.concatenate((x1,x2),axis=0)
yPoints = np.concatenate((y1,y2),axis=0)
xmin = np.min(xPoints)
print(xmin)
xmax = np.max(xPoints)
print(xmax)

histogram = np.sum(binary_warped[int(3*binary_warped.shape[0]/4):,:], axis=0)
xHist = np.linspace(1,1280,1280,dtype=int)
plt.plot(xHist,histogram)


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
# Set the width of the windows +/- margin
windowWidth = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

#for window in range(nwindows):
window=0

leftBottom = int(binary_warped.shape[0]-window*windowHeight) 
leftTop = int(binary_warped.shape[0]-(window+1)*windowHeight)
leftLeft = int(leftx_current-windowWidth/2)
leftRight = int(leftx_current+windowWidth/2)

print(leftBottom, leftTop, leftLeft, leftRight)


out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

cv2.rectangle(out_img,(leftLeft,leftTop),(leftRight,leftBottom-1),
    (0,255,0), 4) 

leftWindow = binary_warped[leftTop:leftBottom,leftLeft:leftRight]
leftNPoints = np.sum(leftWindow)
print(leftNPoints)
#if(leftNPoints>minpix):
leftx_current = np.int(np.mean(np.nonzero(leftWindow)[0])) + leftLeft
print(leftx_current)
plt.axvline(x=leftx_current,color='red')
#
#plt.imshow(leftWindow)  ## ZOOM to the windows
#plt.show()

rightBottom = int(binary_warped.shape[0]-window*windowHeight )
rightTop = int(binary_warped.shape[0]-(window+1)*windowHeight)
rightLeft = int(rightx_current-windowWidth/2)
rightRight = int(rightx_current+windowWidth/2)

cv2.rectangle(out_img,(rightLeft,rightTop),(rightRight,rightBottom-1),
    (0,255,0), 4) 

rightWindow = binary_warped[rightTop:rightBottom,rightLeft:rightRight]
rightNPoints = np.sum(rightWindow)

if(rightNPoints>minpix):
    rightx_current = np.int(np.mean(np.nonzero(rightWindow)[0])) + rightLeft
    print(rightx_current)
    plt.axvline(x=rightx_current,color='red')
    
plt.imshow(out_img)
plt.show()       
    
def findGuided():
    

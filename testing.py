# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:26:33 2017

@author: Toshiharu
"""
from imageProcessing import perspectiveCal,Calibration,birdEye
import cv2
import matplotlib.pyplot as plt
import numpy as np

Calibration(rows=6,cols=9,imagesFolder='camera_cal',show=False)

correction = 4
points_orig = np.float32([[(200, 720), (575+correction, 457), (715-correction,457), (1150, 720)]])
points_world = np.float32([[(440, 720), (440, 0), (950, 0), (950, 720)]])

perspectiveCal(points_orig,points_world,directory='camera_cal')



###########################################################################################
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

############################################################################################
# Do camera calibration given object points and image points
img = cv2.imread('camera_cal/calibration1.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('camera_cal/test_undist.jpg',dst)

img = cv2.imread('test_images/straight_lines1.jpg')
dst = birdEye(img, mtx, dist,M)
cv2.imwrite('test_images/straight_line1_unwarped.jpg',dst)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()

import numpy as np
import cv2
import glob

images = []
#images = glob.glob('test_images/straight_lines*.jpg')
images = glob.glob('test_images/challenge/challenge*.jpg')
# Step through the list and search for chessboard corners
from imageFilter import Multifilter

for idx, fname in enumerate(images):
    print(fname)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    top_down = birdEye(img, mtx, dist,M)
    filtered_img = Multifilter(top_down,s_thresh=(200, 255),b_thresh=(145,200),l_thresh=(210,255),sxy_thresh=(30,100), draw=True)
    filtered_rgb = np.dstack((filtered_img, filtered_img, filtered_img))*255
    #cv2.imwrite(fname[:-4]+'_out.jpg',filtered_rgb)
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
#    f.tight_layout()
#    ax1.imshow(top_down)
#    ax1.set_title('Original Image', fontsize=15)
#    ax2.imshow(filtered_img)
#    ax2.set_title('Undistorted and Warped Image', fontsize=15)
#    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.) 
#    histogram = np.sum(filtered_img[filtered_img.shape[0]//2:,:], axis=0)
#    plt.show()
#    plt.plot(histogram)
    plt.show()
#    print(np.sum(filtered_img))



#test1 = cv2.imread('test_images/test5.jpg')
##top_down = birdEye(test1, mtx, dist,M)
##Multifilter(top_down,s_thresh=(210, 255),b_thresh=(145,200),l_thresh=(220,255),sxy_thresh=(20,100), show=False)
## Overlay text
#number = 150.151561
#
#font                   = cv2.FONT_HERSHEY_SIMPLEX
#bottomLeftCornerOfText = (50,50)
#fontScale              = 2
#fontColor              = (255,255,255)
#lineType               = 3
#text = '{}{:.2f}{}'.format('R curvature: ',number, ' m') 
#
#cv2.putText(test1,text, 
#    bottomLeftCornerOfText, 
#    font, 
#    fontScale,
#    fontColor,
#    lineType)
#
#font                   = cv2.FONT_HERSHEY_SIMPLEX
#bottomLeftCornerOfText = (50,50+80)
#fontScale              = 2
#fontColor              = (255,255,0)
#lineType               = 3
#text = '{}{:.2f}{}'.format('R curvature: ',number, ' m') 
#
#cv2.putText(test1,text, 
#    bottomLeftCornerOfText, 
#    font, 
#    fontScale,
#    fontColor,
#    lineType)
#
#plt.imshow(test1)
#plt.show()
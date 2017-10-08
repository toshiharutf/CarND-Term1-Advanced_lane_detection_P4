# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 23:19:34 2017

@author: Toshiharu
"""

from findLines import find_window_centroids, window_mask, fitLines, Line

from imageProcessing import perspectiveCal,Calibration,birdEye

from drawingMethods import drawRegion

import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


import numpy as np
import cv2
import glob, os

images = []
#images = glob.glob('test_images/straight_lines*.jpg')
images = glob.glob('test_images/test*.jpg')
# Step through the list and search for chessboard corners
from imageFilter import Multifilter

inputFolder = 'test_images'
outputFolder = 'output_images'

for file in os.listdir(inputFolder):
    if file.endswith(".jpg"):
#        image = mpimg.imread(imagesFolder+"/"+file)
        image = cv2.imread(inputFolder+"/"+file)

        leftCurve = Line()
        rightCurve = Line()
        
        leftCurve.x_factor = 3.7/480 #% m/pixel
        rightCurve.x_factor = 3.7/480
        
        leftCurve.y_factor= 3/100 # m/pixel
        rightCurve.y_factor = 3/100 # m/pixel
        
        print(file)
        img = cv2.cvtColor(cv2.imread(inputFolder+"/"+file), cv2.COLOR_BGR2RGB)
        top_down = birdEye(img, mtx, dist,M)
        filtered_img = Multifilter(top_down,s_thresh=(210, 255),b_thresh=(155,200),l_thresh=(220,255),sxy_thresh=(30,100), draw=False)
        fitLines(leftCurve,rightCurve,filtered_img, window_width=50, window_height=120, margin=50,max_offset=60, max_Roffset=500, draw = True)
        overImage = drawRegion(img,leftCurve,rightCurve,Minv)
        
        cv2.imwrite(outputFolder+"/"+file[:-4]+'-output.jpg',cv2.cvtColor(overImage, cv2.COLOR_RGB2BGR))
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        f.tight_layout()
        ax1.imshow(top_down)
        ax1.set_title('Bird Eye Image', fontsize=15)
        ax2.imshow(overImage)
        ax2.set_title('Original Image with overlayed region', fontsize=15)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.) 
               
        plt.show()
        print(leftCurve.radius,'m','  ',rightCurve.radius,'m')
        print(rightCurve.line_base_pos-leftCurve.line_base_pos )
        print( np.absolute((rightCurve.line_base_pos+leftCurve.line_base_pos)/2 -img.shape[1]/2)*leftCurve.x_factor, 'm' )



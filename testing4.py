# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 23:19:34 2017

@author: Toshiharu
"""
from pipeLine import laneFindInit
#from findLines import find_window_centroids, window_mask, fitLines, Line
from findLines3 import lineFullSearch, lineSearchGuided, Line

from imageProcessing import perspectiveCal,Calibration,birdEye

from drawingMethods import drawRegion

from imageFilter import Multifilter

import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

calibration = 'camera_cal/calibration.p'
perspective = 'camera_cal/perspective.p'

mtx,dist,M,Minv = laneFindInit(calibration,perspective)

def pipeLine(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    top_down = birdEye(img, mtx, dist,M)
    binary_warped = Multifilter(top_down,s_thresh=(210, 255),b_thresh=(155,200),l_thresh=(220,255),sxy_thresh=(30,100), draw=False)
    #fitLines(leftCurve,rightCurve,binary_warped, window_width=50, window_height=120, margin=50,max_offset=60, max_Roffset=500, draw = False)
    if(leftLine.detected==True):
        lineSearchGuided(binary_warped,leftLine,rightLine,margin = 100,maxOffset=500,minOffset=200,debug=False)
    else:
        lineFullSearch(binary_warped,leftLine,rightLine,nwindows = 9, windowWidth=100, minpix=50,debug = False)    
    overImage = drawRegion(img,leftLine,rightLine,Minv)
    
    return overImage

inputFolder = 'test_images'
outputFolder = 'output_images'

import os

for file in os.listdir(inputFolder):
    if file.endswith(".jpg"):

        leftLine = Line()
        rightLine = Line()
        
        leftLine.x_factor = 3.7/480 #% m/pixel
        rightLine.x_factor = 3.7/480
        
        leftLine.y_factor= 3/100 # m/pixel
        rightLine.y_factor = 3/100 # m/pixel
        
        img = cv2.imread(inputFolder+"/"+file)
        overImage = pipeLine(img)
        
        plt.imshow(overImage)
               
        plt.show()
        
        print(leftLine.radius,'m','  ',rightLine.radius,'m')
        print(rightLine.basePos()-leftLine.basePos() )
        print( ((rightLine.basePos()+leftLine.basePos())/2 -img.shape[1]/2)*leftLine.x_factor, 'm' )




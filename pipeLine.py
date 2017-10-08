# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 02:07:56 2017

@author: Toshiharu
"""

from findLines import find_window_centroids, window_mask, fitLines, Line

from imageProcessing import perspectiveCal,Calibration,birdEye

from drawingMethods import drawRegion

from imageFilter import Multifilter

import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load camera lens distortion correction parameters
import pickle

def laneFindInit(calFile,persFile):
    
    #Load calibration parameters
    dist_pickle = pickle.load( open( calFile, "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    # Load perpective matrices
    dist_pickle = pickle.load( open( persFile, "rb" ) )
    M = dist_pickle["M"]
    Minv = dist_pickle["Minv"]
    
    return mtx,dist,M,Minv



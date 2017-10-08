# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:59:19 2017

@author: Toshiharu
"""

def perspectiveCal(points_orig,points_world,directory):
# Calibrate perspective transformation
# Four points on test image. Hand picked from testimage1.jpg
  
    M = cv2.getPerspectiveTransform(points_orig, points_world)
    Minv = cv2.getPerspectiveTransform(points_world, points_orig)
    
    import pickle  
    perspective_pickle = {}
    perspective_pickle["M"] = M
    perspective_pickle["Minv"] = Minv
    pickle.dump( perspective_pickle, open( directory+'/perspective.p', "wb" ) )
    print('Saved perspective transformation matrices in:'+ directory+'/perspective.p' )
    
def Calibration(rows=6,cols=9,imagesFolder='camera_cal',show=True):
    import numpy as np
    import cv2
    import glob
    import matplotlib.pyplot as plt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    #imagesFolder = 'camera_cal'
    # Step through the list and search for chessboard corners
    images = glob.glob(imagesFolder+'/calibration*.jpg')
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols,rows), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            if show==True:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (cols,rows), corners, ret)
                plt.imshow(img)
                plt.show()
    
    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    import pickle
    
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( imagesFolder+"/calibration.p", "wb" ) )
    print('Saved camera calibration parameters in: '+imagesFolder+'/calibration.p'  )
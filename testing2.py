# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:14:21 2017

@author: Toshiharu
"""

from findLines import find_window_centroids, window_mask, fitLines, Line

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2

# Read in a thresholded image
leftCurve = Line()
rightCurve = Line()

leftCurve.x_factor = 3.7/438 #% m/pixel
rightCurve.x_factor = 3.7/438

leftCurve.y_factor= 3/100 # m/pixel
rightCurve.y_factor = 3/100 # m/pixel

warped = mpimg.imread('output_images/test3_out.jpg')
plt.imshow(warped,cmap='gray')
plt.show
# window settings
window_width = 50 
window_height = 120 # Break image into 6 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

fitLines(leftCurve,rightCurve,warped, window_width=50, window_height=120, margin=100,max_offset=60, max_Roffset=500, draw = True)
print(leftCurve.radius, rightCurve.radius)



window_centroids = find_window_centroids(warped, window_width, window_height, margin)

# If we found any window centers
if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
    
    #Points of the center of each centroid
    left_centroids=np.zeros((len(window_centroids),2))
    right_centroids=np.zeros((len(window_centroids),2))
    y_offset = window_height/2

    # Go through each level and draw the windows 	
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(warped,window_centroids[level][0],level, window_width,window_height)
        r_mask = window_mask(warped,window_centroids[level][1],level, window_width,window_height)
        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
                
        height_coordinate = warped.shape[0]-y_offset-window_height*level
                
        left_centroids[level][0] = window_centroids[level][0]
        left_centroids[level][1] = height_coordinate
                
        right_centroids[level][0] = window_centroids[level][1]
        right_centroids[level][1] = height_coordinate

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
# If no window centers found, just display orginal road image
else:
    output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    
# Fit a second order polynomial to each
left_fit  = np.polyfit(left_centroids[:,1] , left_centroids[:,0], 2)
right_fit = np.polyfit(right_centroids[:,1], right_centroids[:,0], 2)   

ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Display the final results
left_fit2 = np.array([-2.45535714e-04,   2.68214286e-01,   4.10576786e+02-100])
right_fit2 = np.array([ -1.43849206e-04,   2.43095238e-01,   8.50503571e+02-100])
left_fitx2 = left_fit2[0]*ploty**2 + left_fit2[1]*ploty + left_fit2[2]
right_fitx2 = right_fit2[0]*ploty**2 + right_fit2[1]*ploty + right_fit2[2]

plt.imshow(output)
plt.plot(left_fitx, ploty, color='red')
plt.plot(right_fitx, ploty, color='red')
print(left_fit[2]-right_fit[2])

#plt.plot(left_fitx2, ploty, color='blue')
#plt.plot(right_fitx2, ploty, color='blue')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.title('window fitting results')
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:10:57 2017

@author: Toshiharu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

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
    
#    def fitLine(self):
#        return  np.polyfit(self.fitPoints[:,1] , self.fitPoints[:,0], 2)
        
    def linePlot(self):
        ploty = np.linspace(0,self.y_limit-1, self.y_limit )
        plotx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        return plotx,ploty
    
    
# definir output en metros?
        
        

def window_mask(img_ref, center,level,width=50, height=80):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width=50, window_height=80, margin=100):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

#############################################################################################
def sanityCheck(leftCurve,rightCurve,max_offset, max_Roffset):

    if (leftCurve.best_fit.size == 0):
        leftCurve.best_fit = leftCurve.current_fit
        rightCurve.best_fit = rightCurve.current_fit
    
    left_offset = np.absolute(leftCurve.best_fit[2] - leftCurve.current_fit[2])
    right_offset = np.absolute( rightCurve.best_fit[2] - rightCurve.current_fit[2])
    
    if( (left_offset>max_offset) | (right_offset>max_offset) ):
        leftCurve.detected  = False
        rightCurve.detected = False
    else:
        leftCurve.detected  = True
        rightCurve.detected = True
        
    

#############################################################################################

def fitLines(leftCurve,rightCurve,warped, window_width=50, window_height=120, margin=100,max_offset=60, max_Roffset=500, draw = False):
   
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    # If we found any window centers
    if len(window_centroids) > 0:
        if(draw==True):
            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped)
        
        #Points of the center of each centroid
        left_centroids=np.zeros((len(window_centroids),2))
        right_centroids=np.zeros((len(window_centroids),2))
        y_offset = window_height/2
    
        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            if(draw==True):
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

            
        # Fit a second order polynomial to each
        leftCurve.current_fit  = left_fit  = np.polyfit(left_centroids[:,1] , left_centroids[:,0], 2)
        rightCurve.current_fit = right_fit = np.polyfit(right_centroids[:,1], right_centroids[:,0], 2)
        
        sanityCheck(leftCurve,rightCurve,max_offset, max_Roffset)
    
        if( (leftCurve.detected == True) & (rightCurve.detected == True) ):  #if the curve detected is similar to the last detected, average both for smoothing
             leftCurve.best_fit = (leftCurve.current_fit + leftCurve.best_fit)/2
             rightCurve.best_fit = (rightCurve.current_fit + rightCurve.best_fit)/2
             
        else:
             leftCurve.best_fit = (leftCurve.current_fit*0.7 + leftCurve.best_fit*0.3)  # Else, we rely more on the new values.
             rightCurve.best_fit = (rightCurve.current_fit*0.7 + rightCurve.best_fit*0.3)
             
             
        
        #Find the curve parameters in meters
        leftCurveFit_world   = np.polyfit(left_centroids[:,1]*leftCurve.y_factor , left_centroids[:,0]*leftCurve.x_factor, 2)
        rightCurveFit_world  = np.polyfit(right_centroids[:,1]*rightCurve.y_factor, right_centroids[:,0]*rightCurve.x_factor, 2)
        
        y_eval = warped.shape[0]/2*leftCurve.y_factor  # evaluate at the middle of the figure
        leftCurve.radius = ((1 + (2*leftCurveFit_world[0]*y_eval + leftCurveFit_world[1])**2)**1.5) / np.absolute(2*leftCurveFit_world[0])
        rightCurve.radius =  ((1 + (2*rightCurveFit_world[0]*y_eval + rightCurveFit_world[1])**2)**1.5) / np.absolute(2*rightCurveFit_world[0])
        
        # Base position of each line, to find the offset from the lane center
        bottom_y = warped.shape[0]-1
        leftCurve.line_base_pos = leftCurve.best_fit[0]*bottom_y**2 + leftCurve.best_fit[1]*bottom_y + leftCurve.best_fit[2]
        rightCurve.line_base_pos = rightCurve.best_fit[0]*bottom_y**2 + rightCurve.best_fit[1]*bottom_y + rightCurve.best_fit[2]
        
#        leftCurve.fitPoints  = left_centroids
#        rightCurve.fitPoints = right_centroids
#        leftCurve.fitLine(left_centroids[:,1] , left_centroids[:,0])
#        rightCurve.fitLine(right_centroids[:,1], right_centroids[:,0])

        if(draw==True):
            # Draw the results
#            ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
#            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            left_fitx,ploty = leftCurve.linePlot()
            right_fitx,ploty = rightCurve.linePlot()
            
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
            
            plt.imshow(output)
            plt.plot(left_fitx, ploty, color='red')
            plt.plot(right_fitx, ploty, color='red')
            
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.title('window fitting results')
            plt.show()
            

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        

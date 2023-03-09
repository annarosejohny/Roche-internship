# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 09:48:11 2022

@author: johnya
"""
import cv2
import numpy as np
import glob
import re

def FillingLevelDetector(img_level,
                         max_filling_level):
    
    indicator_position = retrieve_indicatior_position(img_level)
    filling_level = round(indicator_position[0] / max_filling_level * 100, 2)
    full_flag = filling_level > 99.9
   
    return indicator_position, filling_level, full_flag


def retrieve_indicatior_position(bw_image):

    pos_array = np.argwhere((bw_image > 0))
    indicator_position = np.median(pos_array, axis=0).astype(int)

    return indicator_position


def create_mask_from_blank(img_blank):

    img_blank_cropped = img_blank

    img_blank_gray = cv2.cvtColor(img_blank_cropped, cv2.COLOR_BGR2GRAY)
    gauss = cv2.adaptiveThreshold(
        img_blank_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    retval2, otsu = cv2.threshold(
        img_blank_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = (otsu == 0) | (gauss == 0)
    max_filling_level = extract_max_filling_level(mask)
    
    return max_filling_level


def bool_to_binary(img):
    img = np.invert(img)
    img = img.astype(np.uint8)  # convert to an unsigned byte
    img *= 255 
    
    return img


def extract_max_filling_level(mask):
    
    mask_horizontal = extract_horizontal_lines(
        bool_to_binary(mask), horizontal_div=30)
    
    pos_array = np.argwhere((mask_horizontal > 0))
    
    return pos_array.min(axis=0)[0]


def extract_horizontal_lines(img, horizontal_div=30):

    horizontal = np.copy(img)

    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = int(cols // horizontal_div)

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    return horizontal


if __name__ == '__main__':

    filenames = glob.glob(r"C:\Users\JOHNYA\Desktop\python_files\results\level_detector_images\frame_final_*.jpg")
    
    background = [cv2.imread(img) for img in filenames[0:2]]
   
    for back in background:
        
        background_gray = cv2.cvtColor(back, cv2.COLOR_BGR2HSV)[:,:,2]
    
    filenames = sorted(filenames, key=lambda x: float(re.findall(r'\d+', x)[-1]))
   
    images = [cv2.imread(img) for img in filenames]
      
    for img in images:
        
        
        max_filling_level = create_mask_from_blank(img)
        
        frame = img
        org_img = img
        scale_percent = 30 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
          
        # resize image
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        background_gray = cv2.resize(background_gray, dim, interpolation = cv2.INTER_AREA)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,2]
        
        # find the difference between current frame and base frame
        frame_diff = cv2.absdiff(gray,background_gray)
        
        scale = 1
        delta = 0
        ddepth = cv2.CV_8U
         
        grad_x = cv2.Sobel(frame_diff, ddepth, 0,1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        
        
        # # # thresholding to convert the frame to binary
        # ret, thres = cv2.threshold(grad_x, 50, 255, cv2.THRESH_BINARY)
        
        # # # # dilate the frame a bit to get some more white area...
        # # # # # ... makes the detection of contours a bit easier
        # # dilate_frame = cv2.dilate(thres, None, iterations=2)
        horizontal = extract_horizontal_lines(abs_grad_x, horizontal_div=150)
        
    
        # # find the contours around the white segmented areas
        # contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # # draw the contours, not strictly necessary
  
        # for i, cnt in enumerate(contours):
        #     cv2.drawContours(frame, contours, i, (0, 0, 255), 3)

       
        #     if cv2.contourArea(cnt) < 500:
        #       continue
        #     (x, y, w, h) = cv2.boundingRect(cnt)
         
        #     if filling_level-30 >= 100:
        #         text1 = 'FULL' +"{:<15}".format(filling_level-30)
        #         # get the xmin, ymin, width, and height coordinates from the contours
        #         cv2.putText(org_img, text1, (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        #         # draw the bounding boxes
        #         cv2.rectangle(org_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #     else:
        #         text2 = "{:<15}".format(filling_level-30)
        #         cv2.putText(org_img, text2 , (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        #         # draw the bounding boxes
        #         cv2.rectangle(org_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
          
            # cv2.rectangle(org_img, , (x+w, y+h), (0, 255, 0), 2)
        # cv2.line(resized, (0,0),(indicator_position), (0,255,0), thickness=2, lineType=8)
        cv2.imshow('Detected Objects', np.hstack((resized, horizontal)))
        cv2.waitKey(500)
        
    cv2.destroyAllWindows()

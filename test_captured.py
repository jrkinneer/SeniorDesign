import numpy as np
import cv2
from kmeans import kmeans_multiple_grayscale
import os
import time

#create combination of color with gaussian blur 
loop_index = 0  

COLOR = "_blue"
INPUT_PATH = "captured_images" + COLOR
 
for filename in os.listdir(INPUT_PATH):
    start = time.time()
    
    f = os.path.join(INPUT_PATH, filename)
    
    s = filename.split("_")
    s2 = s[1].split(".")
    
    img_index = int(s2[0])
    #input image
    img = cv2.imread(f)
    
    contrast = 5. # Contrast control ( 0 to 127)
    brightness = 2. # Brightness control (0-100)

    # call addWeighted function. use beta = 0 to effectively only operate on one image
    out = cv2.addWeighted( img, contrast, img, 0, brightness)

    #split into color channels
    red = out[:,:,0]
    green = out[:,:,1]
    blue = out[:,:,2]
    
    #run k means to get binary segmentation
    blue_segmented, blue_means, ids = kmeans_multiple_grayscale(blue, 2, 10, 10)
    
    #run k means to get binary segmentation
    green_segmented, green_means, ids = kmeans_multiple_grayscale(green, 2, 10, 10)
    
    #run k means to get binary segmentation
    red_segmented, red_means, ids = kmeans_multiple_grayscale(red, 2, 10, 10)
    
    grey_segmented, grey_means, ids = kmeans_multiple_grayscale(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), 2, 10, 10)
    
    #get the color of the darkest of the two binary segments
    darkest_red = np.min(red_means)
    darkest_green = np.min(green_means)
    darkest_blue = np.min(blue_means)
    darkest_grey = np.min(grey_means)
    
    #initial all black masks
    red_mask = np.zeros_like(img)
    green_mask = np.zeros_like(img)
    blue_mask = np.zeros_like(img)
    grey_mask = np.zeros_like(img)
    
    #blurs red portion of img
    blurred = cv2.GaussianBlur(out[:,:,0], (7,7), 0)
    
    #run k means to get binary segmentation
    blurred_segmented, blurred_means, ids = kmeans_multiple_grayscale(blurred, 2, 10, 10)
    
    darkest_blur = np.min(blurred_means)
    
    blurred_mask = np.zeros_like(img)
    final_mask_blur = np.zeros_like(img)
    final_mask_color = np.zeros_like(img)
    final_mask_grey = np.zeros_like(img)
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if red_segmented[x][y] == darkest_red:
                red_mask[x][y] = img[x][y]
                
            if green_segmented[x][y] == darkest_green:
                green_mask[x][y] = img[x][y]
                
            if blue_segmented[x][y] == darkest_blue:
                blue_mask[x][y] = img[x][y]
                
            if blue_segmented[x][y] == darkest_blue or green_segmented[x][y] == darkest_green or red_segmented[x][y] == darkest_red:
                final_mask_color[x][y] = img[x][y]
                
            if blurred_segmented[x][y] == darkest_blur:
                blurred_mask[x][y] = img[x][y]
                
            if blurred_segmented[x][y] == darkest_blur or red_segmented[x][y] == darkest_red:
                final_mask_blur[x][y] = img[x][y]
            if grey_segmented[x][y] == darkest_grey:
                final_mask_grey[x][y] = img[x][y]
    
    cv2.imwrite("results"+COLOR+"/masks/blurred/img_"+str(img_index)+".png", blurred_mask)
    cv2.imwrite("results"+COLOR+"/masks/combined/img_"+str(img_index)+".png", final_mask_blur)
    cv2.imwrite("results"+COLOR+"/masks/red/img_"+str(img_index)+".png", red_mask)
    cv2.imwrite("results"+COLOR+"/masks/blue/img_"+str(img_index)+".png", blue_mask)
    cv2.imwrite("results"+COLOR+"/masks/green/img_"+str(img_index)+".png", green_mask)
    cv2.imwrite("results"+COLOR+"/masks/combined_rgb/img_"+str(img_index)+".png", final_mask_color)
    cv2.imwrite("results"+COLOR+"/masks/rgb_to_grey/img_"+str(img_index)+".png", final_mask_grey)
    
    cv2.imwrite("results"+COLOR+"/segments/blurred/img_"+str(img_index)+".png", blurred_segmented)
    cv2.imwrite("results"+COLOR+"/segments/red/img_"+str(img_index)+".png", red_segmented)
    cv2.imwrite("results"+COLOR+"/segments/blue/img_"+str(img_index)+".png", blue_segmented)
    cv2.imwrite("results"+COLOR+"/segments/green/img_"+str(img_index)+".png", green_segmented)
    cv2.imwrite("results"+COLOR+"/segments/rgb_to_grey/img_"+str(img_index)+".png", grey_segmented)

    
    stop = time.time()
    print("finished loop iteration ", loop_index, "for first loop, time elapsed = ", stop-start)
    
    loop_index = loop_index + 1
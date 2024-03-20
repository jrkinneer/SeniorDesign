import numpy as np
import cv2
from kmeans import kmeans_multiple_grayscale

def create_mask(img):
    #prepare image for segmentation
    contrast = 5
    brightness = 2
    
    adjusted = cv2.addWeighted(img, contrast, img, 0, brightness)
    
    #split to color channels
    red = adjusted[:,:,1]
    green = adjusted[:,:,2]
    blue = adjusted[:,:,2]
    
    #kmeans on every color channel
    #run k means to get binary segmentation
    blue_segmented, blue_means, _ = kmeans_multiple_grayscale(blue, 2, 10, 10)
    
    #run k means to get binary segmentation
    green_segmented, green_means, _ = kmeans_multiple_grayscale(green, 2, 10, 10)
    
    #run k means to get binary segmentation
    red_segmented, red_means, _ = kmeans_multiple_grayscale(red, 2, 10, 10)

    #get the darkest of each segment
    darkest_red = np.min(red_means)
    darkest_green = np.min(green_means)
    darkest_blue = np.min(blue_means)

    #initial all black masks
    red_mask = np.zeros_like(img)
    green_mask = np.zeros_like(img)
    blue_mask = np.zeros_like(img)
    
    final_mask_color = np.zeros_like(img)
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if blue_segmented[x][y] == darkest_blue or green_segmented[x][y] == darkest_green or red_segmented[x][y] == darkest_red:
                final_mask_color[x][y] = img[x][y]
                
    return final_mask_color
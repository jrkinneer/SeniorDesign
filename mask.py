import cv2
import numpy as np
from tqdm import tqdm

def mask(original_img, line_img):
    line_color = np.array([[255,0,0]])
    masked = np.zeros_like(original_img)
    x = 0
    y = 0
    
    mask = False
    #go row by row through image
    while (x < original_img.shape[0]):
        
        y = 0
        mask = False
        while (y < original_img.shape[1]):
            
            #if we hit a pixel that is the same color as the line's we've drawn we start creating mask
            if line_img[x][y][0] == 255:
                masked[x][y] = original_img[x][y]
                mask = not mask
                y = y + 1
                continue
            if mask:
                masked[x][y] = original_img[x][y]
                
            y = y+1
            
        x = x+1
               
    return masked     
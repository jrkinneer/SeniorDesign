import cv2
import numpy as np
import os

clicked_points = []

def get_click(event, x, y, flags, params):
    global clicked_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

def get_ground_truth(clicked_points, original_img, img_ind):
    
    lines = np.zeros((original_img.shape[0], original_img.shape[1]))
    ind = 0
    while ind < len(clicked_points):
        if ind == len(clicked_points) - 1:
            point1 = clicked_points[ind]
            point2 = clicked_points[0]
            
        else:
            point1 = clicked_points[ind]
            point2 = clicked_points[ind+1]
                
        lines = cv2.line(lines, point1, point2, 255, 1)
                
        ind = ind + 1
            
    cv2.imshow("lines for mask", lines)
    cv2.waitKey()
    cv2.destroyAllWindows()
         
    filled = np.zeros_like(lines)
    
    for i in range(filled.shape[0]):
        # inOutline = False
        j = 0
        
        j2 = original_img.shape[1] - 1
        
        while lines[i][j] != 255 and j < original_img.shape[1] - 1:
            j = j + 1
            
        while lines[i][j2] != 255 and j2 >= 0:
            j2 = j2 - 1
            
        while j < j2:
            filled[i][j] = 255
            j = j + 1
    
    cv2.imshow("mask", filled)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite("ground_truth_blue/img_"+str(img_ind)+".png", filled)
            
        
if __name__ =="__main__":
    for filename in os.listdir("captured_images_blue"):
        clicked_points = []
        
        f = os.path.join('captured_images_blue', filename)
    
        s = filename.split("_")
        s2 = s[1].split(".")
        
        img_index = int(s2[0])
        #input image
        img = cv2.imread(f)
    
        cv2.imshow("image", img)
        
        cv2.setMouseCallback('image', get_click)
        
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        
        get_ground_truth(clicked_points, img, img_index)
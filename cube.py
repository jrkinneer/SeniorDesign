import numpy as np
import cv2
import pyrealsense2 as rs
import time
import qrCode as qr  

def cubeLocator(rmat, tvec, s=qr.QR_DIMENSION, d=qr.D, c=qr.CUBE):
    """_summary_

    Args:
        projected_points (_type_): the projected points gathered from finding the qr code
        rvec (_type_): rotation of the qr code
        tvec (_type_): translation of the qr code
    """
    #see layout.png for visual of coordinate system
    
    #XYZ homogenous coordinates of top face of the cube, relative to qr code origin
    cube_cords = np.array([[s+d, 0, -c, 1],
                        [s+d+c, 0, -c, 1],
                        [s+d+c, c, -c, 1],
                        [s+d, c, -c, 1]]).reshape((4,1,4)).astype('float32')
    
    #get rotation and translation matrix
    essential = np.hstack((rmat, tvec))
    
    pixel_cords = np.zeros((4,2))
    
    ind = 0
    for cord in cube_cords:
        camera = np.dot(qr.INTRINSIC, essential)
        pixel = np.dot(camera, cord.T)
        pixel_cords[ind] = [int(pixel[0]/pixel[2]), int(pixel[1]/pixel[2])]
        
        ind += 1
     
    #world coordinates with the top left corner of the cube as the origin
    cube_cords_origin = np.array([[0, 0, 0],
                        [c, 0, 0],
                        [c, c, 0],
                        [0, c, 0]]).reshape((4,1,3)).astype('float32')   
    
    success, rvec, tvec = cv2.solvePnP(cube_cords_origin, pixel_cords.astype('float32'), qr.INTRINSIC, qr.DIST)
    
    if success:
            
        #turn rotation vector into 3x3 matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        #find the x y and z displacement of the four corners of the qr code, relative to the camera center
        corner_cords = []
        for corner in cube_cords_origin:
            world_corner = np.dot(rmat.T, corner.reshape((3,1))) + tvec
            corner_cords.append(world_corner.flatten())
        
        return corner_cords, pixel_cords, rvec*(180/np.pi), tvec, rmat

    return [], pixel_cords, [], [], []
    
def drawCubeMask(color_image, top_face_cords, rmat, tvec, c=qr.CUBE):
    masked_cube = np.zeros_like(color_image)
    
    blank = np.zeros((color_image.shape[0], color_image.shape[1]))
    
    bottom_cords_homogenous = np.array([[0, 0, c, 1],
                                        [c, 0, c, 1],
                                        [c, c, c, 1],
                                        [0, c, c, 1]]).reshape((4,1,4)).astype('float32')
    
    #get rotation and translation matrix
    essential = np.hstack((rmat, tvec))
    
    pixel_cords = np.zeros((4,2))
    
    ind = 0
    for cord in bottom_cords_homogenous:
        camera = np.dot(qr.INTRINSIC, essential)
        pixel = np.dot(camera, cord.T)
        pixel_cords[ind] = [int(pixel[0]/pixel[2]), int(pixel[1]/pixel[2])]
        
        ind += 1
    
    qr.showBox(blank, top_face_cords)
    qr.showBox(blank, pixel_cords)
    
    #draw lines connecting top face to bottom face
    for i in range(pixel_cords.shape[0]):
        cv2.line(blank, (int(pixel_cords[i][0]), int(pixel_cords[i][1])), (int(top_face_cords[i][0]), int(top_face_cords[i][1])), 255, 1)
    
    for i in range(blank.shape[0]):
        j = 0
        
        j2 = blank.shape[1] - 1
        
        while blank[i][j] != 255 and j < blank.shape[1] - 1:
            j = j + 1
            
        while blank[i][j2] != 255 and j2 >= 0:
            j2 = j2-1
            
        while j < j2:
            masked_cube[i][j] = color_image[i][j]
            j = j + 1
    
    return masked_cube

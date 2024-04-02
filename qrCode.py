import numpy as np
import cv2
import pyrealsense2 as rs

#side length of qr code in milimeters 
QR_DIMENSION = 170.5

#side length of cube in milimeters
CUBE = 38
#distance between cube and qr code
D = 10

#these values are determined by running the calibrate_camera.py script
INTRINSIC = np.array([[608.90339309, 0, 321.37995736],
                      [0, 608.25294592, 244.79138807],
                      [0,0,1]])

DIST = np.array([[-1.66935674e-02, 1.06924906e+00, 3.68738069e-04, -1.45018584e-03, -4.06331275e+00]])

#finds QR code and returns its translation and rotaion relative to the camera
def qrCodeDetect(frame):
    """_summary_
        detects a qrCode in a frame, returns the rotation, translation, and projection points
        the projected points are the four points around the code, used to create a bounding box
    Args:
        frame (NDArray): input image in the form of a numpy array

    Returns:
        corner_cords NDArray: XYZ coordinates in mm of the four corners of the qr code relative to the camera center
        points NDArray: contains points necessary to project three rgb axis onto the captured image
        rvec NDArray: rotation of QR code relative to camera coordinate system (vector, in degrees)
        tvec NDArray: translation of QR code relative to camera coordinate system (vector)
        rmat NDArray: rvec in 3x3 matrix form (in radians)
    """
    #qr code detector object
    qr = cv2.QRCodeDetector()

    #find qr code
    code, points = qr.detect(frame)
    
    #qr code found successfully
    if code:
        #reshape points array for ease of use later
        points = points[0]
        
        #world coordinates of qr code corners assuming no rotation
        #qr_size is measure in mm
        qr_corners = np.array([[0,0,0],
                        [QR_DIMENSION,0,0],
                        [QR_DIMENSION,QR_DIMENSION,0],
                        [0,QR_DIMENSION,0]]).reshape((4,1,3)).astype('float32')
        
        success, rvec, tvec = cv2.solvePnP(qr_corners, points.astype('float32'), INTRINSIC, DIST)
        
        #conditional return depending on the success of PnP solve
        if success:
            
            #turn rotation vector into 3x3 matrix
            rmat, _ = cv2.Rodrigues(rvec)
            
            #find the x y and z displacement of the four corners of the qr code, relative to the camera center
            corner_cords = []
            for corner in qr_corners:
                world_corner = np.dot(rmat.T, corner.reshape((3,1))) + tvec
                corner_cords.append(world_corner.flatten())
            
            return corner_cords, points, rvec*(180/np.pi), tvec, rmat

        else:
            return [], [], [], [], []

    else:
        return [], [], [], [], []

#functions for debugging and visualization purposes
def showCrosshairs(color_image):
    """takes input image and draws crosshairs for demonstration purposes

    Args:
        color_image (NDArray): input image that is altered to have blue crosshairs after the function returns
    """
    cv2.line(color_image, (0, int(color_image.shape[0]/2)), (color_image.shape[1], int(color_image.shape[0]/2)), (255,0,0), 2)
    cv2.line(color_image, (int(color_image.shape[1]/2), 0), (int(color_image.shape[1]/2), color_image.shape[0]), (255,0,0), 2)
    return
    
def showBox(image, points):
    """Show bounding box around a square region of space in the input image

    Args:
        image (NDArray): image array passed by value
        points (NDArray): 4x2 array of u,v pixel coordinates that mark the four corners of the bounding box
    """
    center = (int(points[0][0]), int(points[0][1]))
    limit = 5*image.shape[1]
    
    #error checks to see if the center coord is past an acceptable limit
    if center[0] > limit or center[1] > limit:
        return
    #draws bouding box on qr code
    ind = 0
    while ind < points.shape[0]:
        if ind == points.shape[0] - 1:
            point1 = (int(points[ind][0]), int(points[ind][1]))
            point2 = (int(points[0][0]), int(points[0][1]))
            
        else:
            point1 = (int(points[ind][0]), int(points[ind][1]))
            point2 = (int(points[ind+1][0]), int(points[ind+1][1]))
        
        #error checks to see if the center coord is past an acceptable limit
        if point1[0] > limit or point1[1] > limit:
            break
        
        #checks to see if image is black and white or a color and draws accordingly
        if image.ndim == 3:
            cv2.line(image, point1, point2, (0,255,0), 3)
        else:
            cv2.line(image, point1, point2, 255, 1)
        ind = ind + 1

def showCorners(color_image, points):
    """draws red, green, yellow and pink circles on the corners of a bounding box

    Args:
        color_image (NDArray): input image array passed by value
        points (NDArray): 4x2 numpy array containing u,v pixel coordinates of the four corners
    """
    cv2.circle(color_image, (int(points[0][0]), int(points[0][1])), 4, (0,0,255), 3)
    cv2.circle(color_image, (int(points[1][0]), int(points[1][1])), 4, (0,255,0), 3)
    cv2.circle(color_image, (int(points[2][0]), int(points[2][1])), 4, (255,0,255), 3)
    cv2.circle(color_image, (int(points[3][0]), int(points[3][1])), 4, (0,255,255), 3)
    
def showCordsAndRotation(color_image, points, rvec):
    """displays the XYZ displacement in mm of four points in the input image, as well as the rotation of points with respect to the camera

    Args:
        color_image (NDArray): input image array passed by value
        points (NDArray): 4x2 array containing u,v pixel coordinates
        rvec (NDArray): 3x1 vector containing information about rotation on the x,y,z axis in degrees
    """
    #display coords top left
    s = str(round(points[0][0])) + " " + str(round(points[0][1])) + " " + str(round(points[0][2]))
    cv2.putText(color_image, s, (int(color_image.shape[1]/2)+2, 20), fontFace=1, fontScale=1, color=(0,0,255), thickness=2)
    
    #display coords of top right
    s = str(round(points[1][0])) + " " + str(round(points[1][1])) + " " + str(round(points[1][2]))
    cv2.putText(color_image, s, (int(color_image.shape[1]/2)+2, 40), fontFace=1, fontScale=1, color=(0,255,0), thickness=2)
    
    #display coords of bottom right
    s = str(round(points[2][0])) + " " + str(round(points[2][1])) + " " + str(round(points[2][2]))
    cv2.putText(color_image, s, (int(color_image.shape[1]/2)+2, 60), fontFace=1, fontScale=1, color=(255,0,255), thickness=2)
    
    #display coords bottom left
    s = str(round(points[3][0])) + " " + str(round(points[3][1])) + " " + str(round(points[3][2]))
    cv2.putText(color_image, s, (int(color_image.shape[1]/2)+2, 80), fontFace=1, fontScale=1, color=(0,255,255), thickness=2)
    
    #show rotation info
    cv2.putText(color_image, str(np.round(rvec)), (int(color_image.shape[1]/8), 20), fontFace=1, fontScale=1, color=(0,0,255), thickness=2)
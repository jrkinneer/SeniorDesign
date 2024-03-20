import numpy as np
import cv2
import glob
import time

#Square size in milimeters
SIZE = 19

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)

#add to points to account for chessboard square size
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)*SIZE
#objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('images/calibration_images/*.png')

print("getting images")
for fname in images:
    img = cv2.imread(fname)
    print(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,7), corners2, ret)
        cv2.imshow("show", img)
        cv2.waitKey(500)

    img1=img
    
cv2.destroyAllWindows()

print(">==> Starting calibration")
ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#print(ret)
print("Camera Matrix")
print(cam_mtx)

print("Distortion Coeff")
print(dist)

print("r vecs")
print(rvecs[2])

print("t Vecs")
print(tvecs[2])

print(">==> Calibration ended")
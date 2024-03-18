import numpy as np
import cv2
import pyrealsense2 as rs

#side length of qr code in milimeters 
QR_DIMENSION = 170.5
#these values are determined by running the calibrate_camera.py script
INTRINSIC = np.array([[608.90339309, 0, 321.37995736],
                      [0, 608.25294592, 244.79138807],
                      [0,0,1]])

DIST = np.array([[-1.66935674e-02, 1.06924906e+00, 3.68738069e-04, -1.45018584e-03, -4.06331275e+00]])

#finds QR code and returns its translation and rotaion relative to the camera
def qrCodeDetect(frame, qr_size):
    """_summary_
        detects a qrCode in a frame, returns the rotation, translation, and projection points
        the projected points are the four points around the code, used to create a bounding box
    Args:
        frame (NDArray): input image in the form of a numpy array

    Returns:
        corner_cords NDArray: XYZ coordinates in mm of the four corners of the qr code relative to the camera center
        projected_points NDArray: contains points necessary to project three rgb axis onto the captured image
        rvec NDArray: rotation of QR code relative to camera coordinate system (vector)
        tvec NDArray: translation of QR code relative to camera coordinate system (vector)
        rmat NDArray: rvec in 3x3 matrix form
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
                        [qr_size,0,0],
                        [qr_size,qr_size,0],
                        [0,qr_size,0]]).reshape((4,1,3)).astype('float32')
        
        success, rvec, tvec = cv2.solvePnP(qr_corners, points.astype('float32'), INTRINSIC, DIST)
        
        #conditional return depending on the success of PnP solve
        if success:
            
            #turn rotation vector into 3x3 matrix
            rmat, _ = cv2.Rodrigues(rvec)
            
            #adjust translation on z axis accordingly
            #tvec[2] = tvec[2] + 1.88 + qr_size
            
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
  
def cubeLocator(rmat, tvec, s, d, c):
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
        camera = np.dot(INTRINSIC, essential)
        pixel = np.dot(camera, cord.T)
        pixel_cords[ind] = [int(pixel[0]/pixel[2]), int(pixel[1]/pixel[2])]
        
        ind += 1
     
    #world coordinates with the top left corner of the cube as the origin
    cube_cords_origin = np.array([[0, 0, 0],
                        [c, 0, 0],
                        [c, c, 0],
                        [0, c, 0]]).reshape((4,1,3)).astype('float32')   
    
    success, rvec, tvec = cv2.solvePnP(cube_cords_origin, pixel_cords.astype('float32'), INTRINSIC, DIST)
    
    if success:
            
        #turn rotation vector into 3x3 matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        #adjust translation on z axis accordingly
        #tvec[2] = tvec[2] + 1.88 + qr_size
        
        #find the x y and z displacement of the four corners of the qr code, relative to the camera center
        corner_cords = []
        for corner in cube_cords_origin:
            world_corner = np.dot(rmat.T, corner.reshape((3,1))) + tvec
            corner_cords.append(world_corner.flatten())
        
        return corner_cords, pixel_cords, rvec*(180/np.pi), tvec, rmat

    return [], pixel_cords, [], [], []

#functions for debugging and visualization purposes
def showCrosshairs(color_image):
    cv2.line(color_image, (0, int(color_image.shape[0]/2)), (color_image.shape[1], int(color_image.shape[0]/2)), (255,0,0), 2)
    cv2.line(color_image, (int(color_image.shape[1]/2), 0), (int(color_image.shape[1]/2), color_image.shape[0]), (255,0,0), 2)
    return
    
def showBox(color_image, points):
    center = (int(points[0][0]), int(points[0][1]))
    limit = 5*color_image.shape[1]

    #draws bouding box on qr code
    ind = 0
    while ind < points.shape[0]:
        if ind == points.shape[0] - 1:
            point1 = (int(points[ind][0]), int(points[ind][1]))
            point2 = (int(points[0][0]), int(points[0][1]))
            
        else:
            point1 = (int(points[ind][0]), int(points[ind][1]))
            point2 = (int(points[ind+1][0]), int(points[ind+1][1]))
        
        if center[0] > limit or center[1] > limit:
            break
        if point1[0] > limit or point1[1] > limit:
            break
        
        cv2.line(color_image, point1, point2, (0,255,0), 3)
        ind = ind + 1

def showCorners(color_image, points):
    cv2.circle(color_image, (int(points[0][0]), int(points[0][1])), 4, (0,0,255), 3)
    cv2.circle(color_image, (int(points[1][0]), int(points[1][1])), 4, (0,255,0), 3)
    cv2.circle(color_image, (int(points[2][0]), int(points[2][1])), 4, (255,0,255), 3)
    cv2.circle(color_image, (int(points[3][0]), int(points[3][1])), 4, (0,255,255), 3)
    
def showCordsAndRotation(color_image, qr_cords, rvec):
    #display coords top left
    s = str(round(qr_cords[0][0])) + " " + str(round(qr_cords[0][1])) + " " + str(round(qr_cords[0][2]))
    cv2.putText(color_image, s, (int(color_image.shape[1]/2)+2, 20), fontFace=1, fontScale=1, color=(0,0,255), thickness=2)
    
    #display coords of top right
    s = str(round(qr_cords[1][0])) + " " + str(round(qr_cords[1][1])) + " " + str(round(qr_cords[1][2]))
    cv2.putText(color_image, s, (int(color_image.shape[1]/2)+2, 40), fontFace=1, fontScale=1, color=(0,255,0), thickness=2)
    
    #display coords of bottom right
    s = str(round(qr_cords[2][0])) + " " + str(round(qr_cords[2][1])) + " " + str(round(qr_cords[2][2]))
    cv2.putText(color_image, s, (int(color_image.shape[1]/2)+2, 60), fontFace=1, fontScale=1, color=(255,0,255), thickness=2)
    
    #display coords bottom left
    s = str(round(qr_cords[3][0])) + " " + str(round(qr_cords[3][1])) + " " + str(round(qr_cords[3][2]))
    cv2.putText(color_image, s, (int(color_image.shape[1]/2)+2, 80), fontFace=1, fontScale=1, color=(0,255,255), thickness=2)
    
    #show rotation info
    cv2.putText(color_image, str(np.round(rvec)), (int(color_image.shape[1]/8), 20), fontFace=1, fontScale=1, color=(0,0,255), thickness=2)
                   
#main function for debugging
if __name__ == "__main__":
    #set up RealSense Camera
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            
            # Validate that both frames are valid
            # if not aligned_depth_frame or not color_frame:
            #     continue
            
            #depth_image = np.asanyarray(aligned_depth_frame.get_data()) 
            color_image = np.asanyarray(color_frame.get_data())

            qr_cords, projected, rvec, tvec = qrCodeDetect(color_image, QR_DIMENSION)
            
            #error check to see if qrCodeDetect runs successfully
            if len(projected) > 0:
                projected = projected.reshape((4, 2))

                center = (int(projected[0][0]), int(projected[0][1]))
                limit = 5*color_image.shape[1]

                #loops through the three axis from projected points and the three colors
                ind = 0
                while ind < projected.shape[0]:
                    if ind == projected.shape[0] - 1:
                        point1 = (int(projected[ind][0]), int(projected[ind][1]))
                        point2 = (int(projected[0][0]), int(projected[0][1]))
                        
                    else:
                        point1 = (int(projected[ind][0]), int(projected[ind][1]))
                        point2 = (int(projected[ind+1][0]), int(projected[ind+1][1]))
                    
                    if center[0] > limit or center[1] > limit:
                        break
                    if point1[0] > limit or point1[1] > limit:
                        break
                    
                    cv2.line(color_image, point1, point2, (0,255,0), 3)
                    ind = ind + 1
                
            cv2.imshow('image', color_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        pipeline.stop()

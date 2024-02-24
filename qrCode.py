import numpy as np
import cv2
import pyrealsense2 as rs

FILEPATH = "intrinsics_HD.txt"

def cameraParams(filepath):
    #from text file containing focal length and optical center in the format:
    #fx, fy
    #cx, cy
    data = np.genfromtxt(filepath, delimiter=",")
    
    #[fx, 0, cx],
    #[0, fy, cy],
    #[0,  0,  1]]
    intrinsics = np.array([[data[0][0], 0, data[1][0]],
                           [0, data[0][1], data[1][1]],
                           [0,          0,         1]])
    
    #assuming 0 distortion
    dist = np.zeros((1,5))
    
    return intrinsics, dist
  
#finds QR code and returns its translation and rotaion relative to the camera
def qrCodeDetect(frame):
    """_summary_

    Args:
        frame (NDArray): input image in the form of a numpy array

    Returns:
        projected_points NDArray: contains points necessary to project three rgb axis onto the captured image
        rvec NDArray: rotation of QR code relative to camera coordinate system
        tvec NDArray: translation of QR code relative to camera coordinate system
    """
    #get camera params
    intrinsics, dist = cameraParams(FILEPATH)
    
    #qr code detector object
    qr = cv2.QRCodeDetector()
    
    #find qr code
    code, points = qr.detect(frame)
    
    #qr code found
    if code:
        #corners of the QR code
        edges = np.array([[0,0,0],
                        [0,1,0],
                        [1,1,0],
                        [1,0,0]]).reshape((4,1,3)).astype('float32')
        
        success, rvec, tvec = cv2.solvePnP(edges, points.astype('float32'), intrinsics, dist)
        #cv2.solvePnP(edges, points, intrinsics, dist)
        
        #points for the rgb axis projected onto the image for debugging purposes
        unit_points = np.array([[0,0,0],
                                [1,0,0],
                                [0,1,0],
                                [0,0,1]]).reshape((4,1,3)).astype('float64')
        
        #conditional return depending on the success of PnP solve
        if success:
            projected_points, _ = cv2.projectPoints(unit_points, rvec, tvec, intrinsics, dist)
            return projected_points, rvec, tvec
            
        else:
            return [], [], []
        
    else:
        return [], [], []

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
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

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
            color_frame = aligned_frames.get_color_frame()

            color_image = np.asanyarray(color_frame.get_data())
            
            projected, rvec, tvec = qrCodeDetect(color_image)
            
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            
            #error check to see if qrCodeDetect runs successfully
            if len(projected) > 0:
                projected = projected.reshape((4, 2))
                
                center = (int(projected[0][0]), int(projected[0][1]))
                
                #limit for projection
                limit = 5*color_image.shape[1]
                
                #loops through the three axis from projected points and the three colors
                for p, c in zip(projected[1:], colors[:3]):
                    #reformat p to be an even integer for indexing
                    p = (int(p[0]), int(p[1]))
                    
                    #error check for out of bounds 
                    
                    if center[0] > limit or center[1] > limit:
                        break
                    if p[0] > limit or p[1] > limit:
                        break
                    
                    cv2.line(color_image, center, p, c, 5)
                    
            cv2.imshow('image', color_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            
    finally:
        pipeline.stop()
    
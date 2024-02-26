import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt

FILEPATH = "intrinsics.txt"

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
        detects a qrCode in a frame, returns the rotation, translation, and projection points
        the projected points are the four points around the code, used to create a bounding box
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

        #points for the line projection onto the qr code, (bounding square)
        unit_points = np.array([[0,0,0],
                                [1,0,0],
                                [1,1,0],
                                [0,1,0]]).reshape((4,1,3)).astype('float64')

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
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # rX = []
    # rY = []
    # rZ = []
    
    # tX = []
    # tY = []
    # tZ = []
    
    # x = []
    # x_i = 0
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

            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 100, 0)]

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

                # rX.append(rvec[0][0])
                # rY.append(rvec[1][0])
                # rZ.append(rvec[2][0])
                
                # tX.append(tvec[0][0])
                # tY.append(tvec[2][0])
                # tZ.append(tvec[1][0])
                
                # x.append(x_i)
                # x_i += 1
                
            cv2.imshow('image', color_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                
                # plt.plot(x, rX, label="rotationX", color="red")
                # plt.plot(x, rY, label="rotationY", color="blue")
                # plt.plot(x, rZ, label="rotationZ", color="green")
                # plt.plot(x, tX, label="transX", color="red")
                # plt.plot(x, tY, label="transY", color="blue")
                # plt.plot(x, tZ, label="transZ", color="green")
                # plt.legend()
                # plt.show()
                break

    finally:
        pipeline.stop()

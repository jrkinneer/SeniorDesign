import numpy as np
import cv2 
import pyrealsense2 as rs
import os
import qrCode as qr
import cube
from tqdm import tqdm

COLOR = "red"
RAWPATH = "images/raw_images/"+COLOR+"/rgb"
DEPTH_PATH = "images/raw_images/"+COLOR+"/depth"

#SAVE_PATH = "images/training/"+COLOR
SAVE_PATH = "images/training/"

def captureRaw(N):
    """Capture N images to a folder to be later parse to training images
    """
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
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    print("capturing images")
    ind = 0
    try:
        while ind < N:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            path = RAWPATH + "/" + str(ind) + ".png"
            cv2.imwrite(path, color_image)
            
            path = DEPTH_PATH + "/" + str(ind) + ".png"
            cv2.imwrite(path, depth_image)
            
            #show preview of data being captured
            cv2.imshow("preview", color_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            ind = ind + 1
            
    finally:
        print("image capture finished")
        cv2.destroyAllWindows()
        pipeline.stop()
        
def parseRawToTraining(path, starting_index):
    #steps
    #for image in raw folder
        #read image
        #find qr code
        #if found:
            #remove qr code from image
            #save corresponding masked image
            #save pose to text file
        #else:
            #continue

    for filename in os.listdir(path):
        #input rgb image
        f = os.path.join(path, filename)
        img = cv2.imread(f)
        
        #search for qr code
        xyz_cords, qr_pixel_coords, _, tvec, rmat =  qr.qrCodeDetect(img)
        
        #if found
        if len(xyz_cords) > 0:
            #mask qrcode to white
            
            #get cube location in the pciture space
            _, cube_top_pixels, _, cube_tvec, cube_rmat = cube.cubeLocator(rmat, tvec)
            
            #images to save for demo video
            # save = np.copy(img)
            # qr.showBox(save, pixel_points)
            # save2 = np.copy(img)
            # qr.showBox(save2, cube_top_pixels)
            
            
            #save all training data
            
            
    #index_file.close()
    #out.release()
    #cv2.destroyAllWindows()
if __name__ == "__main__":
    colors = ["red", "blue", "green"]
    img_index = 0
    
    
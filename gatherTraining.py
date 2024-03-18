import numpy as np
import cv2 
import pyrealsense2 as rs
import os
import qrCode as qr
import cube
from tqdm import tqdm

COLOR = "blue"
RAWPATH = "images/raw_images/"+COLOR+"/rgb"
DEPTH_PATH = "images/raw_images/"+COLOR+"/depth"

SAVE_PATH = "images/training/"+COLOR

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
            
            ind = ind + 1
            
    finally:
        print("image capture finished")
        pipeline.stop()
        
def parseRawToTraining():
    #steps
    #for image in raw folder
        #read image
        #find qr code
        #if found:
            #mask cube
            #save corresponding masked image
            #save corresponding raw image
            #save corresponding depth image
            #save pose to text file
            #record image_index in text file
        #else:
            #continue
            
    index_file = open(SAVE_PATH+"/index.txt", "a")
    
    for filename in tqdm(os.listdir(RAWPATH)):
        f = os.path.join(RAWPATH, filename)
        
        s = filename.split('.')
        
        #input rgb image
        img = cv2.imread(f)
        
        #search for qr code
        xyz_cords, _, _, tvec, rmat =  qr.qrCodeDetect(img)
        
        #if found
        if len(xyz_cords) > 0:
            #get cube masked
            _, cube_top_pixels, _, cube_tvec, cube_rmat = cube.cubeLocator(rmat, tvec)
            
            masked_cube_img = cube.drawCubeMask(img, cube_top_pixels, cube_rmat, cube_tvec)
        
            #save rgb
            cv2.imwrite(SAVE_PATH+"/rgb/"+filename, img)
            #save depth
            cv2.imwrite(SAVE_PATH+"/depth/"+filename, cv2.imread(os.path.join(DEPTH_PATH, filename)))
            #save rgb mask
            cv2.imwrite(SAVE_PATH+"/mask/"+filename, masked_cube_img)
            #save pose matrix
            pose = np.hstack((rmat, tvec))
            pose = np.vstack((pose, np.array([0,0,0,1])))
            np.savetxt(SAVE_PATH+"/pose/"+s[0]+".txt", pose)
            #append index to text file
            index_file.write(s[0]+"\n")
        
    index_file.close()
        
if __name__ == "__main__":
    captureRaw(100)
    parseRawToTraining()
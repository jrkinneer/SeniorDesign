import numpy as np
import cv2 
import pyrealsense2 as rs
import time
import os
from qrCode import qrCodeDetect

COLOR = "orange"
RAWPATH = "raw_images/"+COLOR
DEPTH_PATH = "depth_images/"+COLOR

def captureRaw():
    """Capture 1200 images to a folder to be later parse to training images
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

    # Streaming loop
    ind = 0
    try:
        while True:
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
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            path = RAWPATH + "/img_" + str(ind) + ".png"
            cv2.imwrite(path, color_image)
            
            path = DEPTH_PATH + "/img_" + str(ind) + ".png"
            cv2.imwrite(path, color_image)
            
            ind = ind + 1
            
            if ind > 1200:
                break 
            
    finally:
        pipeline.stop()
        
def parseRawToTraining():
    #steps
    #for image in raw folder
        #read image
        #find qr code
        #if found:
            #blur code out of picture
            #run masking for block
            #save corresponding masked image
            #save corresponding raw image
            #save corresponding depth image
            #save pose to text file
            #record image_index in text file
        #else:
            #continue
            
    for filename in os.listdir(RAWPATH):
        f = os.path.join(RAWPATH, filename)
    
        s = filename.split("_")
        s2 = s[1].split(".")
        
        img_index = int(s2[0])
        
        #input rgb image
        img = cv2.imread(f)
        
        #search for qr code
        projected, rvec, tvec = qrCodeDetect(img)
        
        #successful qr code detection
        if len(projected) > 0:
            #get 2 dimensional area that needs blurred out to eliminate qr code
            center = (int(projected[0][0]), int(projected[0][1]))
                
            #limit for projection
            limit = 5*img.shape[1]

            lines = []
            successful_projection = True
            #loops through the three axis from projected points
            for p in projected[1:]:
                #reformat p to be an even integer for indexing
                p = (int(p[0]), int(p[1]))
                lines.append(p)
                #error check for out of bounds 
                if center[0] > limit or center[1] > limit:
                    successful_projection = False
                    break
                if p[0] > limit or p[1] > limit:
                    successful_projection = False
                    break
                
            #abort the loop if projection is not successful
            #we do this because if we do not know where to blur the QR code will interfere with the image masking
            if not successful_projection:
                continue
            
            #add two more lines to list to complete a square
            
            qr_mask = np.zeros_like(img)
            
            #draw white lines on black mask to flood fill for masking
        
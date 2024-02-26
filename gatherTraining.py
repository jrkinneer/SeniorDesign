import numpy as np
import cv2 
import pyrealsense2 as rs
import os
from qrCode import qrCodeDetect
import copy
from mask import create_mask

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
        
        qr_mask = np.zeros((img.shape[0], img.shape[1]))
        #successful qr code detection
        if len(projected) > 0:
            #get 2 dimensional area that needs blurred out to eliminate qr code
            center = (int(projected[0][0]), int(projected[0][1]))
                
            #limit for projection
            limit = 5*img.shape[1]

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
                
                cv2.line(qr_mask, point1, point2, 255, 3)
                ind = ind + 1
            
            #for debuggin
            cv2.imshow("lines", qr_mask)
            cv2.waitKey()
            cv2.destroyAllWindows()
            
            code_masked_out = copy.deepcopy(img)
            
            #flood fill qr mask image
            for i in range(qr_mask.shape[0]):
                # inOutline = False
                j = 0
                
                j2 = img.shape[1] - 1
                
                while qr_mask[i][j] != 255 and j < qr_mask.shape[1] - 1:
                    j = j + 1
                    
                while qr_mask[i][j2] != 255 and j2 >= 0:
                    j2 = j2 - 1
                    
                while j < j2:
                    #fill the rgb image with white where the qr code is supposed to be
                    #more efficient than creating a black and white mask then
                    #using a where function to mask over the rgb image
                    code_masked_out[i][j] = (255, 255, 255)
                    j = j + 1

            rgb_mask = create_mask(code_masked_out)
            
            #save data
            #regular rgb
            
        else:
            continue
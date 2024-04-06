from ultralytics import YOLO
import cv2
import numpy

def train():
    
    #load trained model
    model = YOLO('yolov8n-pose.pt')
    
    #train on my data
    #we specify image size with imgsz and allow for rectangular images with rect=True
    #go over all data ten times with epochs=10
    #plots=True saves data about the models performance and metrics at the end
    #pose=14 emphasizes the importance of getting the orientation right, default = 12
    #verbose = True just prints more data about the process to the terminal
    #name, specifies location to save the model to 
    
    results = model.train(data='/home/jared/SeniorDesign/model.yaml', epochs=10, imgsz=(640,480), rect=True,
                          plots = True, verbose=True, pose=14.0, name='first_full_run')
    
    success = model.export(format='onnx', dynamic=True)
    
train()
# model = YOLO("/home/jared/SeniorDesign/runs/pose/first_full_run2/weights/best.pt")

# img = cv2.imread("/home/jared/datasets/images/training/000024.png")
# cv2.imshow("test", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# results = model(img)
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     print(boxes.xywh[3])
#     print(boxes.xywhn)
#     print(boxes.xyxy)
#     print(boxes.xyxyn)
    
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     print(keypoints)
#     probs = result.probs  # Probs object for classification outputs
#     result.show()
#     result.save(filename='result.png')
#print(results)
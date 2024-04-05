from ultralytics import YOLO
import cv2
import numpy

def train():
    
    #load trained model
    model = YOLO('yolov8n-pose.pt')
    
    #train on my data
    results = model.train(data='/home/jared/SeniorDesign/model.yaml', epochs=10, imgsz=(640,480), rect=True,
                          plots = True, workers=32, verbose=True, pose=14.0, name='first_full_run')
    
    success = model.export(format='onnx', dynamic=True)
    
# train()
model = YOLO("/home/jared/SeniorDesign/runs/pose/first_full_run/weights/best.pt")

img = cv2.imread("/home/jared/datasets/images/training/000000.png")
# cv2.imshow("test", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

results = model(img)
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()
    result.save(filename='result.png')
print(results)
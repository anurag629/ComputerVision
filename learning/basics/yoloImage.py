from ultralytics import YOLO
import cv2

def yoloImage(img, model):
    model = YOLO(model)
    results = model(img, show=True)
    cv2.waitKey(0)
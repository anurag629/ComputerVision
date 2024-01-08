from ultralytics import YOLO
import cv2
import cvzone
import math

def yoloCamera(model):
    model = YOLO(model)
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    while True:
        ret, frame = cap.read()
        results = model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # draw box/rec on frame
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # draw corner rect
                cvzone.cornerRect(frame, (x1, y1, x2, y2), 20, rt=0)
                
                # draw prediction
                conf = math.ceil(box.conf[0]*100)/100
                cv2.putText(frame, f'{conf}', (x1+10, y1+20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                
                cvzone.putTextRect(frame, f'{conf}', (x1, y1), 2, 2, offset=10)
                
        
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
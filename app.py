import cv2
import os
from ultralytics import YOLO
import config

# Start the Webcam
stream = cv2.VideoCapture(0)

while cv2.waitKey(1) !=  ord("x"):
    FACE_MODEL = YOLO(os.path.join(config.MODELS_PATH, "yolov8n-face.pt"))

    _, frame = stream.read()
    result = FACE_MODEL(frame, verbose=config.VERBOSE)

    prossessed_frame = result[0].plot() # type: ignore 
    
    cv2.imshow("my window", prossessed_frame) 


stream.release() 
cv2.destroyAllWindows()
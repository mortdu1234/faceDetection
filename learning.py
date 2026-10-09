import cv2
import os
from ultralytics import YOLO
import config


def check_from_image():
    # to find YOLO models : https://github.com/akanametov/yolo-face
    global_model = YOLO(os.path.join(config.MODELS_PATH, "yolov8n.pt"))
    face_model = YOLO(os.path.join(config.MODELS_PATH, "yolov8n-face.pt"))

    photo = cv2.imread(os.path.join(config.IMAGES_PATH, "demo.jpg"))
    # cv2.moveWindow("my window", 1800, 200)
    result = face_model(photo)



    cv2.imshow("my window", result[0].plot()) # type: ignore
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def webcam_video():
    stream = cv2.VideoCapture(0)

    while cv2.waitKey(1) !=  ord("x"):
        _, frame = stream.read()
        
        cv2.imshow("my window", frame) # type: ignore

    cv2.waitKey(5000)
    stream.release() 
    cv2.destroyAllWindows()


webcam_video()
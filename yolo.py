import cv2
from ultralytics import YOLO

def run_yolo(image_path):
    # Load the model
    yolo = YOLO("bone.pt")

    # Load the image file
    image = cv2.imread(image_path)

    # Predict on the image
    res = yolo.predict(source=image, save=False)

    return res


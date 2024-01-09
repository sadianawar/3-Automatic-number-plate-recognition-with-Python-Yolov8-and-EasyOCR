import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
from ultralytics import YOLO
from util_images import read_license_plate, write_csv

results = {}

# Load YOLO model for license plate detection
license_plate_detector = YOLO('G:/Car Number Plate Detection/V3__Automatic number plate recognition with Python, Yolov8 and EasyOCR/models/licence_plate_detector_last_4_100_epoch.pt')

# Load the image
image_path = 'G:/Car Number Plate Detection/DATASETS/test dataset/UK License Plate/car 14.jpg'
frame = cv2.imread(image_path)

# Detect license plates using YOLO
license_plates = license_plate_detector(frame)[0]
for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate

    # Crop license plate
    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

    # Process the license plate
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow('original_crop', license_plate_crop)
    # cv2.imshow('threshold', license_plate_crop_thresh)
    # cv2.waitKey(0)

    # Read license plate number
    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

    if license_plate_text is not None:
        results[class_id] = {'license_plate': {'bbox': [x1, y1, x2, y2],
                                               'text': license_plate_text,
                                               'bbox_score': score,
                                               'text_score': license_plate_text_score}}

# Write results to CSV
write_csv(results, './image_output_test.csv')

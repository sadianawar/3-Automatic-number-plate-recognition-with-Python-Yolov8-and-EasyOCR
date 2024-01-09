# ------------ IMPLEMENTING OVERALL PIPELINE --------------------
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# -------------- LOAD MODELS ------------------
coco_model = YOLO('G:/Car Number Plate Detection/V3__Automatic number plate recognition with Python, Yolov8 and EasyOCR/models/yolov8n.pt')
license_plate_detector = YOLO('G:/Car Number Plate Detection/V3__Automatic number plate recognition with Python, Yolov8 and EasyOCR/models/licence_plate_detector_last_4_100_epoch.pt')

# --------------- LOAD VIDEO --------------------
cap = cv2.VideoCapture('G:\Car Number Plate Detection\V3__Automatic number plate recognition with Python, Yolov8 and EasyOCR\car_video_sample.mp4')
# cap = cv2.VideoCapture('./car_video_sample.mp4')

vehicles = [2, 3, 5, 7]
# 2=car, 3=motorbike, 5=bus, 7=truck

# ---------------- READ FRAMES ----------------------
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # if frame_nmr > 10:
        #     break
        results[frame_nmr] = {}
        # ---------------- DETECT VEHICLES -------------------
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            # print(detection)
            x1, y1, x2, y2, score, class_id = detection
            # x1, y1, x2, y2 are bounding box and confidence score, then class id
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # --------------- TRACK VEHICLES ---------------------
        track_ids = mot_tracker.update(np.asarray(detections_))

        # --------------- DETECT LICENSE PLATES -----------------
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # -------------- ASSIGN LICENCE PLATE TO CAR ---------------
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # --------------- CROP LICENCE PLATE -----------------------
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # --------------- PROCESSING THE LICENCE PLATE -----------------
                # (apply img filter, improve img)
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # cv2.imshow('original_crop', license_plate_crop)
                # cv2.imshow('threshold', license_plate_crop_thresh)
                # cv2.waitKey(0)

                # ------------------ READ LICENCE PLATE NUMBER ---------------------
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# ---------------------- WRITE RESULTS --------------------------
write_csv(results, './output_test.csv')




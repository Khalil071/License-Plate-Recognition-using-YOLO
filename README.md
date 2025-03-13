License Plate Recognition using YOLO

Overview

This project builds a deep learning model using YOLO (You Only Look Once) to detect and recognize car license plates from images and video streams. The model identifies license plates and extracts the text using Optical Character Recognition (OCR).

Features

Detect car license plates in images and videos

Use pre-trained YOLO model for plate detection

Extract text from plates using Tesseract OCR

Support for real-time license plate recognition

Ability to fine-tune the model on custom datasets

Technologies Used

Python

YOLO (You Only Look Once)

OpenCV

PyTorch

TensorFlow/Keras (optional)

Tesseract OCR

Matplotlib/Seaborn

Installation

Clone the repository:

git clone https://github.com/yourusername/license-plate-recognition-yolo.git
cd license-plate-recognition-yolo

Install dependencies:

pip install -r requirements.txt

Install YOLOv5 (if using YOLOv5):

git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

Install Tesseract OCR:

Windows: Download and install from Tesseract OCR

Linux/macOS:

sudo apt install tesseract-ocr  # Ubuntu/Debian
brew install tesseract          # macOS

Model Usage

Detect License Plates in an Image

Load the YOLO model:

from ultralytics import YOLO
import cv2
import pytesseract

model = YOLO("yolov5s.pt")  # Load YOLO model

Process an image:

image_path = "path/to/car_image.jpg"
results = model(image_path)
results.show()  # Display detected license plates

Recognize License Plate Text

Extract plate region and apply OCR:

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = image[y1:y2, x1:x2]  # Crop plate region
        text = pytesseract.image_to_string(plate, config='--psm 7')
        print("Detected Plate Number:", text.strip())

Real-time License Plate Detection

Use a webcam to detect plates in real-time:

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    results.show()
cap.release()

Future Improvements

Train YOLO on a custom dataset for better plate detection

Improve OCR accuracy using deep learning-based text recognition

Deploy as a web-based or mobile application

Implement multi-camera license plate tracking

License

This project is licensed under the MIT License.

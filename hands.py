import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cap.set(4, 720)

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def main():
    while True:
        attempt = 0
        success, img = cap.read()
        while not success and attempt < 5:
            time.sleep(0.5)
            success, img = cap.read()
            attempt += 1
        if not success:
            print("Failed to read frame")
            break
        img = cv2.flip(img, 1)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        results = detector.detect(image)
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for landmark in hand_landmarks:
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord("x"):
            break
    
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
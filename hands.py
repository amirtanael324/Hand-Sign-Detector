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
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
    (5, 9), (9, 13), (13, 17),                # Palm knuckle connections
]

# Landmark names 
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

HAND_COLORS = {
    "Left": (0, 220, 255),   # Cyan
    "Right":  (0, 140, 255),   # Orange
}
CONNECTION_COLORS = {
    "Left": (180, 255, 255), # Light cyan
    "Right":  (100, 200, 255), # Light orange
}

def draw_hand(img, hand_landmarks, label_prefix, point_color, line_color):
    h, w = img.shape[:2]

    # Convert to pixel coordinates
    pts = [
        (int(lm.x * w), int(lm.y * h))
        for lm in hand_landmarks
    ]

    #Draw connections
    for (i, j) in HAND_CONNECTIONS:
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        cv2.line(img, (x1, y1), (x2, y2), line_color, 2, cv2.LINE_AA)

        # Label each connection at its midpoint
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        conn_label = f"{label_prefix}{i}-{label_prefix}{j}"
        cv2.putText(
            img, conn_label, (mx + 3, my - 3),
            cv2.FONT_HERSHEY_PLAIN, 0.55, line_color, 1, cv2.LINE_AA
        )

    #Draw landmark points and labels
    for idx, (x, y) in enumerate(pts):
        cv2.circle(img, (x, y), 6, point_color, -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)  # White border

        pt_label = f"{label_prefix}{idx}"
        cv2.putText(
            img, pt_label, (x + 7, y + 4),
            cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            img, pt_label, (x + 7, y + 4),
            cv2.FONT_HERSHEY_PLAIN, 0.8, point_color, 1, cv2.LINE_AA
        )


def main():
    prev_time = time.time()

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
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                side = handedness[0].display_name  # "Left" or "Right"
                prefix = "L" if side == "Right" else "R"
                opposite = "Left" if side == "Right" else "Right"
                pt_color = HAND_COLORS[opposite]
                ln_color = CONNECTION_COLORS[opposite]

                draw_hand(img, hand_landmarks, prefix, pt_color, ln_color)


        cv2.imshow("Hand Landmarks", img)
        if cv2.waitKey(1) & 0xFF == ord("x"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

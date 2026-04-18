import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import numpy as np

# Camera setup 
cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cap.set(4, 720)

# Download hand landmarker model 
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        model_path
    )

#  MediaPipe setup 
base_options = python.BaseOptions(model_asset_path=model_path)
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector     = vision.HandLandmarker.create_from_options(options)

# Hand drawing config 
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

HAND_COLORS = {
    "Left":  (0, 220, 255),
    "Right": (0, 140, 255),
}
CONNECTION_COLORS = {
    "Left":  (180, 255, 255),
    "Right": (100, 200, 255),
}


def rule_based_asl(hand_landmarks):
    points = np.array([[lm.x, lm.y] for lm in hand_landmarks])
    
    def dist(i, j):
        return np.linalg.norm(points[i] - points[j])
    
    def finger_extended(tip, pip, mcp):
        tip_y, pip_y, mcp_y = points[tip,1], points[pip,1], points[mcp,1]
        return tip_y < pip_y < mcp_y
    
    def finger_curled(tip, mcp):
        return points[tip,1] > points[mcp,1]
    
    # Normalize by palm size
    palm_size = dist(0, 9)
    if palm_size < 0.01: palm_size = 0.1
    
    # Key distances
    d_ti = dist(4, 8) / palm_size
    d_im = dist(8, 12) / palm_size
    d_tm = dist(4, 12) / palm_size
    d_tr = dist(4, 16) / palm_size
    
    # Finger states
    idx_ext = finger_extended(8, 6, 5)
    mid_ext = finger_extended(12, 10, 9)
    rng_ext = finger_extended(16, 14, 13)
    pky_ext = finger_extended(20, 18, 17)
    
    idx_curl = finger_curled(8, 5)
    mid_curl = finger_curled(12, 9)
    rng_curl = finger_curled(16, 13)
    pky_curl = finger_curled(20, 17)
    
    # Thumb state
    thumb_side = points[4,0] > points[3,0] if points[5,0] > points[17,0] else points[4,0] < points[3,0]
    thumb_up = points[4,1] < points[5,1]
    thumb_across_palm = points[4,1] < points[8,1] and points[4,1] < points[12,1]
    
    # WORKING LETTERS 
    if idx_curl and mid_curl and rng_curl and pky_curl and thumb_up:
        return 'A'
    
    if thumb_side and not idx_ext and not mid_ext and not rng_ext and pky_ext:
        return 'Y'
    
    if thumb_side and idx_ext and not mid_ext and not rng_ext and not pky_ext:
        return 'L'
    
    if idx_ext and mid_ext and rng_ext and pky_ext and not thumb_side:
        return 'B'
    
    if idx_ext and mid_ext and rng_ext and not pky_ext and not thumb_side:
        return 'W'
    
    # X, R, H, U, V 
    if not idx_ext and mid_ext and points[8,1] < points[12,1] and points[8,1] < points[9,1]:
        return 'X'
    
    if idx_ext and mid_ext and not rng_ext and not pky_ext and points[12,1] < points[8,1] and d_im < 0.05:
        return 'R'
    
    if idx_ext and mid_ext and rng_curl and pky_curl and d_im < 0.045:
        return 'H'
    
    if idx_ext and mid_ext and rng_curl and pky_curl and d_im < 0.055 and d_im > 0.02:
        return 'U'
    
    if idx_ext and mid_ext and rng_curl and pky_curl and d_im > 0.07:
        return 'V'
    
    # K, P, F, G, Q, D 
    if idx_ext and mid_ext and not rng_ext and not pky_ext and thumb_side:
        return 'K'
    
    if idx_ext and mid_ext and not rng_ext and not pky_ext and points[4,0] > points[8,0] and points[4,0] < points[12,0]:
        return 'P'
    
    if d_ti < 0.07 and mid_ext and rng_ext and pky_ext and points[4,0] < points[8,0]:
        return 'F'
    
    if thumb_side and abs(points[8,1] - points[5,1]) < 0.06 and not idx_ext:
        return 'G'
    
    if not idx_ext and thumb_side and points[8,1] > points[5,1]:
        return 'Q'
    
    if idx_ext and d_tm < 0.09 and not mid_ext:
        return 'D'
    
    # I & J (J = pinky down + left) 
    if not idx_ext and not mid_ext and not rng_ext and pky_ext:
        pinky_tip = points[20]
        pinky_mcp = points[17]
        wrist = points[0]
        
        pinky_up = pinky_tip[1] < pinky_mcp[1]
        pinky_down = pinky_tip[1] > pinky_mcp[1]
        pinky_left = pinky_tip[0] < wrist[0]
        
        if pinky_up:
            return 'I'
        elif pinky_down and pinky_left:
            return 'J'
        else:
            return 'I'
    
    # CURVED/CIRCLED LETTERS 
    if not (idx_ext or mid_ext or rng_ext or pky_ext):
        
        if d_ti < 0.08 and dist(8,12) < 0.08 and dist(12,16) < 0.08:
            return 'O'
        
        elif 0.1 < d_ti < 0.3 and not thumb_across_palm:
            return 'C'
        
        elif thumb_across_palm and points[4,0] > points[12,0] and points[4,0] < points[16,0]:
            return 'T'
        
        elif idx_curl and mid_curl and rng_curl and thumb_across_palm:
            return 'M'
        
        elif idx_curl and mid_curl and thumb_across_palm:
            return 'N'
        
        elif idx_curl and mid_curl and rng_curl and pky_curl and thumb_across_palm:
            return 'E'
        
        elif idx_curl and mid_curl and rng_curl and pky_curl and points[4,1] < points[5,1]:
            return 'S'
    
    return '?'

# Drawing functions 
def draw_hand(img, hand_landmarks, label_prefix, point_color, line_color):
    h, w = img.shape[:2]

    pts = [
        (int(lm.x * w), int(lm.y * h))
        for lm in hand_landmarks
    ]

    for (i, j) in HAND_CONNECTIONS:
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        cv2.line(img, (x1, y1), (x2, y2), line_color, 2, cv2.LINE_AA)
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        conn_label = f"{label_prefix}{i}-{label_prefix}{j}"
        cv2.putText(
            img, conn_label, (mx + 3, my - 3),
            cv2.FONT_HERSHEY_PLAIN, 0.55, line_color, 1, cv2.LINE_AA
        )

    for idx, (x, y) in enumerate(pts):
        cv2.circle(img, (x, y), 6, point_color, -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)

        pt_label = f"{label_prefix}{idx}"
        cv2.putText(
            img, pt_label, (x + 7, y + 4),
            cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            img, pt_label, (x + 7, y + 4),
            cv2.FONT_HERSHEY_PLAIN, 0.8, point_color, 1, cv2.LINE_AA
        )

def draw_asl_label(img, hand_landmarks, label, color):
    h, w = img.shape[:2]
    wrist = hand_landmarks[0]
    wx, wy = int(wrist.x * w), int(wrist.y * h)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness  = 3
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

    badge_x = wx - 20
    badge_y = wy + 20
    pad = 8

    cv2.rectangle(img,
                  (badge_x - pad, badge_y - text_h - pad),
                  (badge_x + text_w + pad, badge_y + pad),
                  (0, 0, 0), -1)
    cv2.rectangle(img,
                  (badge_x - pad, badge_y - text_h - pad),
                  (badge_x + text_w + pad, badge_y + pad),
                  color, 2)
    cv2.putText(img, label, (badge_x, badge_y),
                font, font_scale, color, thickness, cv2.LINE_AA)

# Main loop 
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

        img    = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        results = detector.detect(image)

        if results.hand_landmarks:
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                side     = handedness[0].display_name
                prefix   = "L" if side == "Right" else "R"
                opposite = "Left" if side == "Right" else "Right"
                pt_color = HAND_COLORS[opposite]
                ln_color = CONNECTION_COLORS[opposite]

                draw_hand(img, hand_landmarks, prefix, pt_color, ln_color)

                label = rule_based_asl(hand_landmarks)
                draw_asl_label(img, hand_landmarks, label, pt_color)

        cv2.imshow("Hand Landmarks + ROBUST ASL", img)
        if cv2.waitKey(1) & 0xFF == ord("x"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

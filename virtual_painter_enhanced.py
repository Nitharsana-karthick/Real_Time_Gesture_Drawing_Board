import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# -------------------- Setup --------------------
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# -------------------- Brush Settings --------------------
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # Red, Green, Blue, Yellow
color_index = 0
brush_thickness = 5

# Canvas and strokes for undo
canvas = None
strokes = deque()  # each stroke is a list of points

# Previous point
prev_x, prev_y = None, None
current_stroke = []

# -------------------- Helper Functions --------------------
def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def detect_gesture(hand_landmarks):
    # Simple gestures:
    # 1. Fist (all fingertips below knuckles) -> undo
    # 2. Peace sign (index & middle up) -> clear
    # Returns: 'undo', 'clear', None

    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    finger_up = []

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
            finger_up.append(1)
        else:
            finger_up.append(0)

    if finger_up == [0,0,0,0]:
        return 'undo'
    elif finger_up[:2] == [1,1]:
        return 'clear'
    return None

# -------------------- Main Loop --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # Detect gesture
            gesture = detect_gesture(handLms)
            if gesture == 'clear':
                canvas = np.zeros_like(frame)
                strokes.clear()
                prev_x, prev_y = None, None
                current_stroke = []
            elif gesture == 'undo' and strokes:
                strokes.pop()
                canvas = np.zeros_like(frame)
                for stroke in strokes:
                    for i in range(1, len(stroke)):
                        cv2.line(canvas, stroke[i-1], stroke[i], stroke[0][2], brush_thickness)
                prev_x, prev_y = None, None
                current_stroke = []

            # Get index fingertip
            lm = handLms.landmark[8]
            cx, cy = int(lm.x * w), int(lm.y * h)

            if prev_x is None:
                prev_x, prev_y = cx, cy
                current_stroke = [(cx, cy, colors[color_index])]
            else:
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), colors[color_index], brush_thickness)
                current_stroke.append((cx, cy, colors[color_index]))
                prev_x, prev_y = cx, cy

    else:
        # Hand not detected → save current stroke
        if current_stroke:
            strokes.append(current_stroke)
            current_stroke = []
        prev_x, prev_y = None, None

    # -------------------- Combine Canvas with Camera Feed --------------------
    output = cv2.addWeighted(frame, 0.5, canvas, 1, 0)  # camera + canvas

    # -------------------- Display --------------------
    cv2.putText(output, "Press C to change color | ESC to exit", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Virtual Painter - Camera Background", output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('c'):  # Change brush color
        color_index = (color_index + 1) % len(colors)

cap.release()
cv2.destroyAllWindows()

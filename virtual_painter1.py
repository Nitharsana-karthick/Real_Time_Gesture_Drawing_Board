import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# ----------------------------
# Configuration
# ----------------------------
WIDTH, HEIGHT = 1280, 720
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 70
MAX_UNDO = 20

# ----------------------------
# MediaPipe Setup
# ----------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# ----------------------------
# Variables
# ----------------------------
draw_color = (255, 0, 255)
xp, yp = 0, 0
canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

undo_stack = deque(maxlen=MAX_UNDO)
redo_stack = deque(maxlen=MAX_UNDO)

# ----------------------------
# UI Button Positions
# ----------------------------
buttons = {
    "Pink": ((20, 20), (120, 80), (255, 0, 255)),
    "Blue": ((140, 20), (240, 80), (255, 0, 0)),
    "Green": ((260, 20), (360, 80), (0, 255, 0)),
    "Eraser": ((380, 20), (520, 80), (0, 0, 0)),
    "+": ((540, 20), (600, 80), (200, 200, 200)),
    "-": ((620, 20), (680, 80), (200, 200, 200)),
    "Undo": ((700, 20), (820, 80), (50, 50, 50)),
    "Redo": ((840, 20), (960, 80), (50, 50, 50)),
    "Clear": ((980, 20), (1120, 80), (0, 0, 255)),
    "Save": ((1140, 20), (1260, 80), (0, 255, 255)),
}

# ----------------------------
# Helper Functions
# ----------------------------
def fingers_up(lm):
    fingers = []
    fingers.append(lm[4][0] > lm[3][0])      # Thumb
    fingers.append(lm[8][1] < lm[6][1])      # Index
    fingers.append(lm[12][1] < lm[10][1])    # Middle
    fingers.append(lm[16][1] < lm[14][1])    # Ring
    fingers.append(lm[20][1] < lm[18][1])    # Pinky
    return fingers

def draw_ui(img):
    for name, (p1, p2, color) in buttons.items():
        cv2.rectangle(img, p1, p2, color, cv2.FILLED)
        cv2.putText(img, name, (p1[0]+10, p1[1]+45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def check_button(x, y):
    for name, (p1, p2, _) in buttons.items():
        if p1[0] < x < p2[0] and p1[1] < y < p2[1]:
            return name
    return None

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_ui(img)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            fingers = fingers_up(lm_list)
            x1, y1 = lm_list[8]  # Index finger

            # ---------------- Selection Mode ----------------
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                action = check_button(x1, y1)

                if action:
                    undo_stack.append(canvas.copy())
                    redo_stack.clear()

                    if action == "Pink":
                        draw_color = (255, 0, 255)
                    elif action == "Blue":
                        draw_color = (255, 0, 0)
                    elif action == "Green":
                        draw_color = (0, 255, 0)
                    elif action == "Eraser":
                        draw_color = (0, 0, 0)
                    elif action == "+":
                        BRUSH_THICKNESS += 2
                    elif action == "-":
                        BRUSH_THICKNESS = max(2, BRUSH_THICKNESS - 2)
                    elif action == "Undo" and undo_stack:
                        redo_stack.append(canvas.copy())
                        canvas = undo_stack.pop()
                    elif action == "Redo" and redo_stack:
                        undo_stack.append(canvas.copy())
                        canvas = redo_stack.pop()
                    elif action == "Clear":
                        canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                    elif action == "Save":
                        cv2.imwrite("drawing.png", canvas)

            # ---------------- Drawing Mode ----------------
            elif fingers[1] and not fingers[2]:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                thickness = ERASER_THICKNESS if draw_color == (0,0,0) else BRUSH_THICKNESS
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)
                xp, yp = x1, y1

    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Virtual Painter", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

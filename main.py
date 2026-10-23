import os
import cv2
import math
import random
import time
import numpy as np

# ===================== KONFIGURASI =====================
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
BORDER = 25
HEAD_R = 11
BODY_R = 9
SEG_DIST = 14
SPEED = 16
START_LEN = 5
MIN_LEN = 2
APPLE_COUNT = 3
BOMB_COUNT = 1
ITEM_R = 14
BOMB_LIFETIME = 5.0
APPLE_SCORE = 10
BOMB_SCORE = -10
APPLE_GROW = 1
BOMB_SHRINK = 2
MIN_AREA = 250

white_s_max = 40
white_v_min = 200

click_point = None

def on_mouse(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

def calibrate_white(hsv, pt):
    """Kalibrasi threshold putih dari patch kecil di titik klik."""
    global white_s_max, white_v_min
    x, y = pt
    patch = hsv[max(0, y - 5):y + 6, max(0, x - 5):x + 6].reshape(-1, 3)
    s = int(np.median(patch[:, 1]))
    v = int(np.median(patch[:, 2]))
    white_s_max = int(min(255, s + 35))
    white_v_min = int(max(100, v - 45))

def detect_object(frame):
    """Segmentasi warna -> (centroid, mask). centroid None jika tidak ada."""
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, white_v_min]),
                       np.array([179, white_s_max, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None, mask
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None, mask
    return (m["m10"] / m["m00"], m["m01"] / m["m00"]), mask

def main():
    global click_point, white_s_max, white_v_min
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka. Coba ganti CAM_INDEX.")
        return

    win = "Snake Camera Game - Deteksi"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    show_mask = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        if click_point is not None:
            hsv_full = cv2.cvtColor(cv2.GaussianBlur(frame, (9, 9), 0), cv2.COLOR_BGR2HSV)
            calibrate_white(hsv_full, click_point)
            click_point = None

        pos, mask = detect_object(frame)
        canvas = frame.copy()
        if pos is not None:
            cv2.circle(canvas, (int(pos[0]), int(pos[1])), 14, (0, 255, 0), 3)

        if show_mask:
            cv2.imshow("Mask", mask)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("+"), ord("=")):
            white_v_min = min(255, white_v_min + 5)
        elif key in (ord("-"), ord("_")):
            white_v_min = max(0, white_v_min - 5)
        elif key == ord("]"):
            white_s_max = min(255, white_s_max + 5)
        elif key == ord("["):
            white_s_max = max(0, white_s_max - 5)
        elif key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Mask")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

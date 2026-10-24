import math
import os
import random
import time
import cv2
import numpy as np

BORDER, HEAD_R, BODY_R, ITEM_R = 25, 11, 9, 14
SEG_DIST, SPEED = 14, 16
START_LEN, MIN_LEN = 5, 2
APPLES, BOMBS, BOMB_LIFE = 3, 1, 5.0
MIN_AREA = 250
WIN_CAM, WIN_GAME = "Deteksi Kamera", "Snake Game"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Threshold warna MERAH (hue wrap-around di 0 dan 179)
HUE_TOL = 10
TH = {"h": 0, "s": 100, "v": 70}

ADJUST = {
    ord("+"): ("s", 5), ord("="): ("s", 5),
    ord("-"): ("s", -5), ord("_"): ("s", -5),
    ord("]"): ("v", 5), ord("["): ("v", -5),
}

SPRITES = {}
clicks = []

def to_hsv(frame):
    return cv2.cvtColor(cv2.GaussianBlur(frame, (9, 9), 0), cv2.COLOR_BGR2HSV)

def detect(frame):
    """Segmentasi merah -> (centroid | None, mask)."""
    hsv = to_hsv(frame)
    lo, hi, s, v = TH["h"] - HUE_TOL, TH["h"] + HUE_TOL, TH["s"], TH["v"]

    def rng(a, b):
        return cv2.inRange(hsv, (a, s, v), (b, 255, 255))

    if lo < 0:
        mask = rng(0, hi) | rng(180 + lo, 179)
    elif hi > 179:
        mask = rng(lo, 179) | rng(0, hi - 180)
    else:
        mask = rng(lo, hi)

    k = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(cv2.erode(mask, k, iterations=2), k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        m = cv2.moments(max(cnts, key=cv2.contourArea))
        if m["m00"] >= MIN_AREA:
            return (m["m10"] / m["m00"], m["m01"] / m["m00"]), mask
    return None, mask

def calibrate(frame, pt):
    """Atur threshold warna dari patch kecil di titik klik."""
    x, y = pt
    patch = to_hsv(frame)[max(0, y - 5):y + 6, max(0, x - 5):x + 6].reshape(-1, 3).astype(int)
    TH["h"] = (int(np.median((patch[:, 0] + 90) % 180)) - 90) % 180  # median hue melingkar
    TH["s"] = max(60, int(np.median(patch[:, 1])) - 80)
    TH["v"] = max(40, int(np.median(patch[:, 2])) - 80)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka. Coba ganti CAM_INDEX.")
        return

    cv2.namedWindow(WIN_CAM)
    cv2.moveWindow(WIN_CAM, 20, 40)
    cv2.setMouseCallback(WIN_CAM, on_mouse)

    show_mask = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(cv2.flip(frame, 1), (W, H))

        if clicks:
            calibrate(frame, clicks[-1])
            clicks.clear()

        pos, mask = detect(frame)

        cam = frame.copy()
        if pos:
            c = (int(pos[0]), int(pos[1]))
            cv2.circle(cam, c, 14, (0, 255, 0), 3)
            cv2.drawMarker(cam, c, (0, 255, 0), cv2.MARKER_CROSS, 44, 1)
            cv2.putText(cam, f"Terdeteksi {c}", (10, 25), FONT, 0.65,
                        (0, 0, 0), 3)
            cv2.putText(cam, f"Terdeteksi {c}", (10, 25), FONT, 0.65,
                        (0, 255, 0), 2)
        else:
            cv2.putText(cam, "Objek merah tidak terdeteksi", (10, 25), FONT,
                        0.65, (0, 0, 0), 3)
            cv2.putText(cam, "Objek merah tidak terdeteksi", (10, 25), FONT,
                        0.65, (0, 0, 255), 2)

        cv2.putText(cam, "Klik objek merah = kalibrasi | M = mask | Q = keluar",
                    (10, H - 40), FONT, 0.5, (0, 0, 0), 3)
        cv2.putText(cam, "Klik objek merah = kalibrasi | M = mask | Q = keluar",
                    (10, H - 40), FONT, 0.5, (255, 255, 255), 1)
        info = f"S min: {TH['s']} (+/-)   V min: {TH['v']} ([/])   Hue: {TH['h']}"
        cv2.putText(cam, info, (10, H - 15), FONT, 0.55, (0, 0, 0), 3)
        cv2.putText(cam, info, (10, H - 15), FONT, 0.55, (255, 255, 255), 1)

        cv2.imshow(WIN_CAM, cam)

        if show_mask:
            cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Mask")
        elif key in ADJUST:
            k, d = ADJUST[key]
            TH[k] = min(255, max(0, TH[k] + d))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    

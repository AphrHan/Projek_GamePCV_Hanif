import math
import os
import random
import time
import cv2
import numpy as np
 
CAM_INDEX, W, H = 0, 640, 480
BORDER, HEAD_R, BODY_R, ITEM_R = 25, 11, 9, 14   
SEG_DIST, SPEED = 14, 16                         
START_LEN, MIN_LEN = 5, 2
APPLES, BOMBS, BOMB_LIFE = 3, 1, 5.0             
MIN_AREA = 500                                   
WIN_CAM, WIN_GAME = "Deteksi Kamera", "Snake Game"
FONT = cv2.FONT_HERSHEY_SIMPLEX
 
HUE_TOL = 8                   
TH = {"h": 0, "s": 150, "v": 90}   
ADJUST = {ord("+"): ("s", 10), ord("="): ("s", 10), ord("-"): ("s", -10),
          ord("_"): ("s", -10), ord("]"): ("v", 10), ord("["): ("v", -10)}
SPRITES = {}                  
clicks = []                   
 
# PENGOLAHAN CITRA 
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
    TH["h"] = (int(np.median((patch[:, 0] + 90) % 180)) - 90) % 180   
    TH["s"] = max(110, int(np.median(patch[:, 1])) - 50)     
    TH["v"] = max(60, int(np.median(patch[:, 2])) - 70)
 
# GAMBAR / ASET
def text(img, s, y, scale=0.7, color=(255, 255, 255), thick=2, x=None, box=True):
    """Teks dengan kotak latar gelap (supaya terbaca jelas, tanpa bayangan ganda)."""
    (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
    if x is None:   # rata tengah
        x = (img.shape[1] - tw) // 2
    if box:
        cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
 
def load_sprites():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    for name, size in {"apple": 36, "bomb": 36, "head": 34, "body": 22}.items():
        path = os.path.join(folder, name + ".png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None
        if img is not None:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            SPRITES[name] = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    print("Aset dimuat:", list(SPRITES) or f"tidak ada (cek folder {folder})")
 
def paste(img, name, c):
    """Tempel sprite PNG (dengan transparansi) di titik c."""
    spr = SPRITES[name]
    sh, sw = spr.shape[:2]
    x0, y0 = int(c[0] - sw / 2), int(c[1] - sh / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + sw), min(H, y0 + sh)
    if xa >= xb or ya >= yb:
        return
    part = spr[ya - y0:yb - y0, xa - x0:xb - x0]
    a = part[:, :, 3:4] / 255.0 if part.shape[2] == 4 else 1.0
    roi = img[ya:yb, xa:xb]
    roi[:] = part[:, :, :3] * a + roi * (1 - a)
 
def draw_cam(img, pos):
    """Jendela 1: penanda objek + info threshold."""
    if pos:
        c = (int(pos[0]), int(pos[1]))
        cv2.circle(img, c, 14, (0, 255, 0), 3)
        cv2.drawMarker(img, c, (0, 255, 0), cv2.MARKER_CROSS, 44, 1)
        text(img, f"Terdeteksi {c}", 25, 0.65, (0, 255, 0), x=10)
    else:
        text(img, "Objek merah tidak terdeteksi", 25, 0.65, (0, 0, 255), x=10)
    text(img, "Klik objek merah = kalibrasi | M = mask | Q = keluar", H - 40, 0.5, thick=1, x=10)
    text(img, f"S min: {TH['s']} (+/-)   V min: {TH['v']} ([/])   Hue: {TH['h']}", H - 15, 0.55, thick=1, x=10)
 
# GAME
class Game:
    def __init__(self):
        self.best = 0
        self.reset()
 
    def reset(self):
        self.head, self.path, self.target = [W / 2, H / 2], [], None
        self.length, self.score, self.alive, self.reason = START_LEN, 0, True, ""
        self.apples, self.bombs = [], []             # bom: [x, y, waktu_muncul]
        for _ in range(APPLES):
            self.apples.append(self.spawn())
        for _ in range(BOMBS):
            self.bombs.append([*self.spawn(), time.time()])
 
    def body(self):
        return self.path[::-1][:self.length]
 
    def spawn(self):
        """Posisi acak di dalam dinding, tidak menimpa ular/item lain."""
        m = BORDER + ITEM_R + 8
        busy = [self.head] + self.body() + self.apples + [b[:2] for b in self.bombs]
        for _ in range(100):
            p = (random.randint(m, W - m), random.randint(m, H - m))
            if all(math.dist(p, o) > ITEM_R * 3 for o in busy):
                break
        return p
 
    def die(self, reason):
        self.alive, self.reason = False, reason
 
    def update(self, target):
        self.best = max(self.best, self.score)
        if not self.alive:
            return
        self.target = target
 
        if target:
            d = math.dist(target, self.head)
            if d > 1:
                k = min(SPEED, d) / d
                self.head[0] += (target[0] - self.head[0]) * k
                self.head[1] += (target[1] - self.head[1]) * k
 
        if not self.path or math.dist(self.path[-1], self.head) >= SEG_DIST:
            self.path.append(tuple(self.head))
        del self.path[:-(self.length + 1)]
 
        x, y = self.head
        if not (BORDER + HEAD_R < x < W - BORDER - HEAD_R and
                BORDER + HEAD_R < y < H - BORDER - HEAD_R):
            return self.die("Menabrak dinding!")
 
        if any(math.dist(self.head, p) < HEAD_R + BODY_R - 8 for p in self.body()[3:]):
            return self.die("Menabrak badan sendiri!")
 
        for a in self.apples[:]:
            if math.dist(self.head, a) < HEAD_R + ITEM_R:
                self.apples.remove(a)
                self.length += 1
                self.score += 10
                self.apples.append(self.spawn())
 
        now = time.time()
        for b in self.bombs:
            hit = math.dist(self.head, b[:2]) < HEAD_R + ITEM_R
            if hit:
                self.length = max(MIN_LEN, self.length - 2)
                self.score -= 10
            if hit or now - b[2] >= BOMB_LIFE:
                b[:] = [*self.spawn(), now]
 
    def draw(self, img):
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), (40, 40, 220), BORDER * 2)
        cv2.rectangle(img, (BORDER, BORDER), (W - BORDER, H - BORDER), (255, 255, 255), 2)
 
        for a in self.apples:
            c = (int(a[0]), int(a[1]))
            if "apple" in SPRITES:
                paste(img, "apple", c)
            else:
                cv2.circle(img, c, ITEM_R, (30, 30, 230), -1)
                cv2.ellipse(img, (c[0] + 7, c[1] - ITEM_R - 4), (6, 3), -30, 0, 360, (60, 200, 60), -1)
 
        for b in self.bombs:
            c = (int(b[0]), int(b[1]))
            if "bomb" in SPRITES:
                paste(img, "bomb", c)
            else:
                cv2.circle(img, c, ITEM_R, (30, 30, 30), -1)
                cv2.circle(img, c, ITEM_R, (200, 200, 200), 1)
                cv2.line(img, (c[0] + 5, c[1] - ITEM_R + 2), (c[0] + 10, c[1] - ITEM_R - 6), (0, 180, 230), 2)
            left = max(0.0, BOMB_LIFE - (time.time() - b[2]))
            cv2.ellipse(img, c, (ITEM_R + 6, ITEM_R + 6), -90, 0, 360 * left / BOMB_LIFE, (0, 200, 255), 2)
 
        body = self.body()
        for i in range(len(body) - 1, -1, -1):
            p, t = (int(body[i][0]), int(body[i][1])), i / max(1, len(body))
            if "body" in SPRITES:
                paste(img, "body", p)
            else:
                cv2.circle(img, p, BODY_R, (int(60 + 40 * t), int(200 - 80 * t), int(60 - 30 * t)), -1)
        h = (int(self.head[0]), int(self.head[1]))
        if "head" in SPRITES:
            paste(img, "head", h)
        else:
            cv2.circle(img, h, HEAD_R, (0, 220, 0) if self.alive else (120, 120, 120), -1)
            for dx in (-4, 4):
                cv2.circle(img, (h[0] + dx, h[1] - 3), 2, (255, 255, 255), -1)
 
        text(img, f"Score: {self.score}", BORDER + 28, 0.8, x=BORDER + 8)
        text(img, f"Panjang: {self.length}  Best: {self.best}", BORDER + 55, 0.55, thick=1, x=BORDER + 8)
        if self.alive and self.target is None:
            text(img, "Objek tidak terdeteksi!", H // 2, 0.9, (0, 255, 255))
        if not self.alive:
            img[:] = img * 0.45                      # gelapkan layar
            text(img, "KO!", H // 2 - 50, 3.0, (0, 0, 255), 6, box=False)
            text(img, self.reason, H // 2 + 10, 0.9)
            text(img, f"Score: {self.score}   Best: {self.best}", H // 2 + 50, 0.9, (0, 255, 255))
            text(img, "Tekan R untuk main lagi", H // 2 + 95, 0.8, (200, 200, 200))
 
# MAIN
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
 
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka. Coba ganti CAM_INDEX.")
        return
    for name, x in ((WIN_CAM, 20), (WIN_GAME, W + 50)):
        cv2.namedWindow(name)
        cv2.moveWindow(name, x, 40)
    cv2.setMouseCallback(WIN_CAM, on_mouse)         
 
    load_sprites()
    game, started, show_mask, smooth = Game(), False, False, None
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(cv2.flip(frame, 1), (W, H))      
 
        if clicks:
            calibrate(frame, clicks[-1])
            clicks.clear()
        pos, mask = detect(frame)
 
        if pos is None:
            smooth = None
        else:
            smooth = pos if smooth is None else tuple((a + b) / 2 for a, b in zip(smooth, pos))
 
        cam = frame.copy()
        draw_cam(cam, pos)
 
        canvas = np.full((H, W, 3), 35, np.uint8)
        canvas[::40], canvas[:, ::40] = 45, 45               # grid tipis
        if started:
            game.update(smooth)
            game.draw(canvas)
        else:
            text(canvas, "SNAKE CAMERA GAME", 160, 1.3, (0, 255, 0), 3, box=False)
            text(canvas, "Kalibrasi: klik objek merah di jendela Deteksi Kamera", 230, 0.55, thick=1)
            text(canvas, "Pastikan lingkaran hijau mengikuti objek", 260, 0.55, thick=1)
            text(canvas, "Tekan SPASI untuk mulai", 305, 0.8, (0, 255, 255))
 
        cv2.imshow(WIN_CAM, cam)
        cv2.imshow(WIN_GAME, canvas)
        if show_mask:
            cv2.imshow("Mask", mask)
 
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" ") and not started:
            started = True
            game.reset()
        elif key == ord("r") and not game.alive:
            game.reset()
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

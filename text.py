import pickle
import collections
import cv2
import mediapipe as mp
import numpy as np
import time

# ─── Load model ───────────────────────────────────────────────────────────
model_dict = pickle.load(open('./model.p', 'rb'))
model      = model_dict['model']
scaler     = model_dict.get('scaler')

TOTAL_FEATURES = 63

# ─── MediaPipe ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# ─── Smoothing ────────────────────────────────────────────────────────────
SMOOTH_WINDOW     = 10
prediction_buffer = collections.deque(maxlen=SMOOTH_WINDOW)

# ─── Text-builder state ───────────────────────────────────────────────────
built_text      = ""
last_committed  = ""
current_stable  = ""
stable_start_ts = None
HOLD_SECONDS    = 0.5
SPACE_DELAY     = 0.8
no_hand_since   = None

# ─── Confidence tracking ──────────────────────────────────────────────────
CONF_THRESHOLD = 0.70
current_conf   = 0.0
conf_history   = []          # tích lũy confidence trong suốt hold window

# ─── Layout ───────────────────────────────────────────────────────────────
CAM_W, CAM_H = 640, 480
INFO_W        = 160        # panel thông tin bên phải
TEXT_H        = 70
BORDER        = 2
PAD           = 14

WIN_W = BORDER + CAM_W + BORDER + INFO_W + BORDER   # 806
WIN_H = BORDER + CAM_H + BORDER + TEXT_H + BORDER   # 556

BG    = (18,  18,  18)
PANEL = (28,  28,  28)
WHITE = (230, 230, 230)
GREEN = (80,  200, 120)
AMBER = (0,   165, 255)
RED   = (70,   70, 200)
FONT  = cv2.FONT_HERSHEY_SIMPLEX

WIN_NAME = "ASL - Nhan dien ky hieu tay"

def conf_color(conf: float):
    if conf >= 0.75:
        return GREEN
    if conf >= 0.50:
        return AMBER
    return RED


# ─── Hàm trích xuất đặc trưng (63 features) ──────────────────────────────
def get_distance_features(lm):
    key_pairs = [
        (8, 12), (7, 11), (6, 10),
        (8, 4),  (12, 4),
        (0, 8),  (0, 12), (0, 4),
        (8, 16), (12, 16),
    ]
    ref = np.sqrt((lm[9].x - lm[0].x)**2 + (lm[9].y - lm[0].y)**2)
    ref = max(ref, 1e-6)
    features = []
    for i, j in key_pairs:
        d = np.sqrt((lm[i].x - lm[j].x)**2 + (lm[i].y - lm[j].y)**2)
        features.append(d / ref)
    return np.array(features)


def get_angle_features(lm):
    def angle_3pts(a, b, c):
        v1 = np.array([lm[a].x - lm[b].x, lm[a].y - lm[b].y])
        v2 = np.array([lm[c].x - lm[b].x, lm[c].y - lm[b].y])
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.arccos(np.clip(cos_a, -1.0, 1.0))
    return np.array([
        angle_3pts(6, 7, 8),
        angle_3pts(5, 6, 7),
        angle_3pts(10, 11, 12),
        angle_3pts(9, 10, 11),
        angle_3pts(8, 6, 10),
        angle_3pts(5, 0, 9),
        angle_3pts(2, 3, 4),
        angle_3pts(1, 2, 3),
    ])


def get_cross_finger_features(lm):
    v_index  = np.array([lm[8].x - lm[5].x, lm[8].y - lm[5].y])
    v_middle = np.array([lm[12].x - lm[9].x, lm[12].y - lm[9].y])
    v_index  = v_index  / (np.linalg.norm(v_index)  + 1e-6)
    v_middle = v_middle / (np.linalg.norm(v_middle) + 1e-6)
    return np.array([
        np.dot(v_index, v_middle),
        v_index[0] * v_middle[1] - v_index[1] * v_middle[0],  # cross 2D thủ công
        abs(lm[8].x - lm[12].x),
    ])


def get_enriched_features(hand_landmarks):
    lm = hand_landmarks.landmark
    x_list = [l.x for l in lm]
    y_list = [l.y for l in lm]
    x_min, y_min = min(x_list), min(y_list)
    base = []
    for l in lm:
        base.append(l.x - x_min)
        base.append(l.y - y_min)
    return np.concatenate([
        base,
        get_distance_features(lm),
        get_angle_features(lm),
        get_cross_finger_features(lm),
    ])  # 63 features


# ──────────────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CAM_W, CAM_H))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        no_hand_since = None

        hl       = results.multi_hand_landmarks[0]
        features = get_enriched_features(hl)

        if len(features) == TOTAL_FEATURES:
            feats = features.reshape(1, -1)
            if scaler:
                feats = scaler.transform(feats)

            proba        = model.predict_proba(feats)[0]
            top_idx      = int(np.argmax(proba))
            current_conf = float(proba[top_idx])
            raw_char     = model.classes_[top_idx]

            prediction_buffer.append(raw_char)
            smoothed_char = collections.Counter(prediction_buffer).most_common(1)[0][0]

            # ── Hold logic ────────────────────────────────────────────────
            if smoothed_char != current_stable:
                current_stable  = smoothed_char
                stable_start_ts = time.time()
                conf_history.clear()          # reset khi ký tự thay đổi
            else:
                conf_history.append(current_conf)   # tích lũy mỗi frame
                held = time.time() - stable_start_ts
                min_conf = min(conf_history) if conf_history else 0.0
                if (held >= HOLD_SECONDS
                        and current_stable != last_committed
                        and min_conf >= CONF_THRESHOLD):   # toàn bộ hold phải >= 70%
                    built_text    += current_stable
                    last_committed = current_stable

    else:
        prediction_buffer.clear()
        current_stable  = ""
        stable_start_ts = None
        last_committed  = ""
        current_conf    = 0.0
        conf_history.clear()

        if no_hand_since is None:
            no_hand_since = time.time()
        elif time.time() - no_hand_since >= SPACE_DELAY:
            if built_text and built_text[-1] != " ":
                built_text += " "
            no_hand_since = time.time() + 9999

    # ── Vẽ landmarks lên frame ────────────────────────────────────────────
    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # ── Dựng canvas ───────────────────────────────────────────────────────
    canvas = np.full((WIN_H, WIN_W, 3), 18, dtype=np.uint8)

    # Camera
    cx, cy = BORDER, BORDER
    canvas[cy:cy+CAM_H, cx:cx+CAM_W] = frame
    border_col = conf_color(current_conf) if results.multi_hand_landmarks else (50, 50, 50)
    cv2.rectangle(canvas, (cx-BORDER, cy-BORDER),
                  (cx+CAM_W+BORDER-1, cy+CAM_H+BORDER-1), border_col, BORDER)

    # ── Info panel bên phải ───────────────────────────────────────────────
    ix, iy = BORDER + CAM_W + BORDER, BORDER
    iw, ih = INFO_W, CAM_H
    cv2.rectangle(canvas, (ix, iy), (ix+iw-1, iy+ih-1), PANEL, -1)

    # Tiêu đề nhỏ
    cv2.putText(canvas, "KY TU", (ix+8, iy+22),
                FONT, 0.42, WHITE, 1, cv2.LINE_AA)

    # Ký tự dự đoán (hiển thị lớn)
    ch_disp = current_stable if current_stable else "?"
    ch_col  = conf_color(current_conf) if current_stable else (50, 50, 50)
    (cw, _), _ = cv2.getTextSize(ch_disp, FONT, 3.5, 4)
    cv2.putText(canvas, ch_disp, (ix + (iw - cw) // 2, iy + 115),
                FONT, 3.5, ch_col, 4, cv2.LINE_AA)

    cv2.line(canvas, (ix+8, iy+135), (ix+iw-8, iy+135), (45, 45, 45), 1)

    # Confidence
    conf_col = conf_color(current_conf)
    cv2.putText(canvas, "DO TIN CAY", (ix+8, iy+158),
                FONT, 0.37, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{current_conf*100:.0f}%", (ix+8, iy+183),
                FONT, 0.72, conf_col, 2, cv2.LINE_AA)
    bx1, by1 = ix+8,    iy+193
    bx2, by2 = ix+iw-8, iy+203
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (45, 45, 45), -1)
    fx = bx1 + int((bx2 - bx1) * current_conf)
    if fx > bx1:
        cv2.rectangle(canvas, (bx1, by1), (fx, by2), conf_col, -1)

    # Hold progress
    cv2.putText(canvas, "TIEN TRINH", (ix+8, iy+228),
                FONT, 0.37, WHITE, 1, cv2.LINE_AA)
    px1, py1 = ix+8,    iy+236
    px2, py2 = ix+iw-8, iy+246
    cv2.rectangle(canvas, (px1, py1), (px2, py2), (45, 45, 45), -1)
    if stable_start_ts is not None and results.multi_hand_landmarks:
        held_frac = min(1.0, (time.time() - stable_start_ts) / HOLD_SECONDS)
        hold_col  = (RED if current_conf < CONF_THRESHOLD
                     else (GREEN if held_frac < 1.0 else AMBER))
        fx = px1 + int((px2 - px1) * held_frac)
        if fx > px1:
            cv2.rectangle(canvas, (px1, py1), (fx, py2), hold_col, -1)

    cv2.line(canvas, (ix+8, iy+263), (ix+iw-8, iy+263), (45, 45, 45), 1)

    # Trạng thái
    if not results.multi_hand_landmarks:
        st_txt, st_col = "Khong co tay",  (65, 65, 65)
    elif current_conf < CONF_THRESHOLD:
        st_txt, st_col = "Conf. thap!",   RED
    elif stable_start_ts and (time.time() - stable_start_ts) >= HOLD_SECONDS:
        st_txt, st_col = "Da ghi!",       AMBER
    else:
        st_txt, st_col = "Dang giu...",   GREEN
    cv2.putText(canvas, st_txt, (ix+8, iy+286),
                FONT, 0.40, st_col, 1, cv2.LINE_AA)

    # Phím tắt
    cv2.line(canvas, (ix+8, iy+308), (ix+iw-8, iy+308), (45, 45, 45), 1)
    for i, txt in enumerate(["[Q]  Thoat", "[C]  Xoa het"]):
        cv2.putText(canvas, txt, (ix+8, iy+330 + i*24),
                    FONT, 0.37, WHITE, 1, cv2.LINE_AA)

    # ── Text box ──────────────────────────────────────────────────────────
    ty = BORDER + CAM_H + BORDER
    cv2.rectangle(canvas, (0, ty), (WIN_W-1, ty+TEXT_H-1), PANEL, -1)
    cv2.line(canvas, (0, ty), (WIN_W, ty), (50, 50, 50), 1)

    cv2.putText(canvas, "VAN BAN:", (PAD, ty+16),
                FONT, 0.37, (75, 75, 75), 1, cv2.LINE_AA)

    cursor  = "|" if int(time.time() * 2) % 2 == 0 else " "
    display = built_text + cursor
    max_w   = WIN_W - PAD * 2
    while display:
        (w, _), _ = cv2.getTextSize(display, FONT, 0.85, 2)
        if w <= max_w:
            break
        display = display[1:]
    cv2.putText(canvas, display, (PAD, ty + 52),
                FONT, 0.85, WHITE, 2, cv2.LINE_AA)

    cv2.imshow(WIN_NAME, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 8:
        built_text     = built_text[:-1]
        last_committed = built_text[-1] if built_text else ""
    elif key == ord(' '):
        if built_text and built_text[-1] != " ":
            built_text += " "
    elif key == ord('c'):
        built_text     = ""
        last_committed = ""

cap.release()
cv2.destroyAllWindows()
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
CONF_THRESHOLD = 0.75
current_conf   = 0.0
smoothed_char  = ""

# ─── Layout ───────────────────────────────────────────────────────────────
CAM_W, CAM_H = 640, 480
TEXT_H       = 80
BORDER       = 2
PAD          = 16

WIN_W = CAM_W + BORDER * 2
WIN_H = CAM_H + TEXT_H + BORDER * 3

BG    = (10,  10,  10)
WHITE = (255, 255, 255)
GREEN = (0,  210, 120)
AMBER = (0,  165, 255)
RED   = (60,  60, 200)
GRAY  = (80,  80,  80)
FONT  = cv2.FONT_HERSHEY_SIMPLEX

WIN_NAME = "ASL"

# ─── Word suggestion dictionary ───────────────────────────────────────────
WORD_DICT = [
    # A
    "AN", "ANH", "ANH AY", "AN TOAN", "AM NHAC",
    # B
    "BAN", "BAN BE", "BAO GIO", "BAT DAU", "BUOI SANG", "BUOI TOI",
    # C
    "CAM ON", "CAU HOI", "CHO TOI", "CHUC MUNG", "CO THE", "CON NGUOI",
    # D
    "DA HIEU", "DAY HOC", "DI HOC", "DONG Y", "DU LIEU",
    # G
    "GIA DINH", "GIOI THIEU", "GOI Y", "GUP DO",
    # H
    "HAY NOI", "HOC TAP", "HOI DAP", "HOM NAY", "HOM QUA",
    # K
    "KHO KHAN", "KHONG HIEU", "KHONG BIET", "KY HIEU",
    # L
    "LAM VIEC", "LOP HOC", "LUC NAO",
    # M
    "MO HINH", "MOI NGUOI", "MOT LAN NUA", "MUC TIEU",
    # N
    "NGU PHAP", "NGON NGU", "NGUOI DUNG", "NHAN DANG",
    # P
    "PHAN LOAI", "PHUONG PHAP",
    # Q
    "QUAN TRONG", "QUAY LAI",
    # S
    "SAI ROI", "SIN CHAO", "SUA DOI",
    # T
    "TAP DU LIEU", "THU NGHIEM", "TIEN BO", "TIM HIEU", "TOT LAM",
    "TRAI NGHIEM", "TRO GIUP", "TRUONG HOC",
    # V
    "VAN DE", "VUI LONG", "VUNG TAY",
    # X
    "XIN CHAO", "XIN LOI", "XAC SUAT",
    # Y
    "YEU CAU",
]

suggestions  = []
selected_sug = -1  # index của gợi ý đang được chọn (-1 = không có)
show_hotkeys = False   # toggle hiển thị panel phím tắt

def get_suggestions(text: str, max_count: int = 4) -> list:
    if not text:
        return []
    words  = text.strip().split()
    prefix = words[-1].upper() if words else ""
    if not prefix:
        return []
    return [w for w in WORD_DICT if w.startswith(prefix) and w != prefix][:max_count]

def apply_suggestion(text: str, word: str) -> str:
    words = text.split()
    if words:
        words[-1] = word + " "
        return " ".join(words)
    return word + " "


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
        v_index[0] * v_middle[1] - v_index[1] * v_middle[0],
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
    ])


# ─── Draw helpers ─────────────────────────────────────────────────────────
def draw_hotkeys(canvas, show: bool):
    """Vẽ nút GUIDE góc trên trái. Nếu show=True thì mở panel phím tắt."""
    # ── Nút GUIDE ────────────────────────────────────────────────────────
    btn_x1, btn_y1 = 8, 8
    btn_x2, btn_y2 = 72, 28
    btn_col = (60, 160, 255) if show else (50, 50, 50)
    cv2.rectangle(canvas, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_col, -1)
    cv2.rectangle(canvas, (btn_x1, btn_y1), (btn_x2, btn_y2), (150, 150, 150), 1)
    cv2.putText(canvas, "GUIDE", (btn_x1 + 5, btn_y2 - 6),
                FONT, 0.4, WHITE, 1, cv2.LINE_AA)

    if not show:
        return

    # ── Panel phím tắt ───────────────────────────────────────────────────
    lines = [
        "Q   : Thoat",
        "BSP : Xoa ky tu",
        "SPC : Them khoang trang",
        "C   : Xoa tat ca",
        "TAB : Go y tiep theo",
        "ENT : Ap dung go y",
    ]
    px, py   = 8, 36
    line_h   = 20
    panel_w  = 210
    panel_h  = len(lines) * line_h + 10
    overlay  = canvas.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (px, py), (px + panel_w, py + panel_h),
                  (80, 80, 80), 1)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (px + 8, py + 16 + i * line_h),
                    FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def draw_conf_bar(canvas, conf):
    """Thanh confidence góc trên phải."""
    bw, bh = 120, 14
    x1 = WIN_W - bw - 10
    y1 = 10
    x2, y2 = x1 + bw, y1 + bh
    # label
    cv2.putText(canvas, "CONF", (x1, y1 - 3), FONT, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
    # background
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 80, 80), 1)
    # fill
    fill = x1 + int(bw * conf)
    if fill > x1:
        cv2.rectangle(canvas, (x1, y1), (fill, y2), conf_color(conf), -1)
    # percent text
    cv2.putText(canvas, f"{conf*100:.0f}%", (x2 + 4, y2 - 1),
                FONT, 0.38, WHITE, 1, cv2.LINE_AA)


def draw_current_char(canvas, char, conf, hand_landmarks, frame_w, frame_h):
    """Hiển thị ký hiệu đang nhận dạng ngay trên bounding box bàn tay."""
    if not char or not hand_landmarks:
        return
    lm     = hand_landmarks.landmark
    x_list = [l.x for l in lm]
    y_list = [l.y for l in lm]
    x1 = max(0,      int(min(x_list) * frame_w) - 20) + BORDER
    y1 = max(0,      int(min(y_list) * frame_h) - 20) + BORDER
    x2 = min(frame_w, int(max(x_list) * frame_w) + 20) + BORDER
    y2 = min(frame_h, int(max(y_list) * frame_h) + 20) + BORDER

    # bounding box
    cv2.rectangle(canvas, (x1, y1), (x2, y2), conf_color(conf), 2)

    # ký hiệu lớn bên dưới box
    font_scale, thickness = 2.0, 5
    (tw, th), _ = cv2.getTextSize(char, FONT, font_scale, thickness)
    tx = x1 + (x2 - x1 - tw) // 2
    ty = min(WIN_H - TEXT_H - 10, y2 + th + 8)
    # shadow
    cv2.putText(canvas, char, (tx, ty), FONT, font_scale, (0, 0, 0),    thickness + 3, cv2.LINE_AA)
    cv2.putText(canvas, char, (tx, ty), FONT, font_scale, conf_color(conf), thickness, cv2.LINE_AA)


def draw_suggestions(canvas, suggestions, selected):
    """Hiển thị gợi ý từ ngay phía trên text box."""
    if not suggestions:
        return
    sx      = BORDER + PAD
    sy      = BORDER + CAM_H - 2          # sát trên text box
    box_h   = 22
    box_pad = 8

    for i, word in enumerate(suggestions):
        (tw, _), _ = cv2.getTextSize(word, FONT, 0.5, 1)
        bx1 = sx
        bx2 = sx + tw + box_pad * 2
        by1 = sy - box_h
        by2 = sy - 2

        is_sel  = (i == selected)
        bg_col  = (60, 180, 100) if is_sel else (40, 40, 40)
        txt_col = (0, 0, 0)      if is_sel else (200, 200, 200)

        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg_col, -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (100, 100, 100), 1)
        cv2.putText(canvas, word, (bx1 + box_pad, by2 - 5),
                    FONT, 0.5, txt_col, 1, cv2.LINE_AA)
        sx = bx2 + 6


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

            if smoothed_char != current_stable:
                current_stable  = smoothed_char
                stable_start_ts = time.time()
            else:
                held = time.time() - stable_start_ts
                if (held >= HOLD_SECONDS
                        and current_stable != last_committed
                        and current_conf >= CONF_THRESHOLD):
                    built_text    += current_stable
                    last_committed = current_stable
                    suggestions    = get_suggestions(built_text)
                    selected_sug   = -1

    else:
        prediction_buffer.clear()
        current_stable  = ""
        smoothed_char   = ""
        stable_start_ts = None
        last_committed  = ""
        current_conf    = 0.0

        if no_hand_since is None:
            no_hand_since = time.time()
        elif time.time() - no_hand_since >= SPACE_DELAY:
            if built_text and built_text[-1] != " ":
                built_text  += " "
                suggestions  = get_suggestions(built_text)
                selected_sug = -1
            no_hand_since = time.time() + 9999

    # ── Compose canvas ────────────────────────────────────────────────────
    canvas = np.full((WIN_H, WIN_W, 3), 10, dtype=np.uint8)

    # ── Camera frame ──────────────────────────────────────────────────────
    cx, cy = BORDER, BORDER
    canvas[cy:cy+CAM_H, cx:cx+CAM_W] = frame

    # ── Vẽ khung xương tay (landmarks + connections) ──────────────────────
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                canvas[cy:cy+CAM_H, cx:cx+CAM_W],
                hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # ── Viền camera ───────────────────────────────────────────────────────
    border_col = conf_color(current_conf) if results.multi_hand_landmarks else WHITE
    cv2.rectangle(canvas, (cx - BORDER, cy - BORDER),
                  (cx + CAM_W + BORDER - 1, cy + CAM_H + BORDER - 1), border_col, BORDER)

    # ── Confidence bar (góc trên phải) ────────────────────────────────────
    if results.multi_hand_landmarks:
        draw_conf_bar(canvas, current_conf)

    # ── Hold progress bar ─────────────────────────────────────────────────
    if results.multi_hand_landmarks and stable_start_ts is not None:
        held_frac = min(1.0, (time.time() - stable_start_ts) / HOLD_SECONDS)
        bx1, by1 = cx + 10,       cy + CAM_H - 22
        bx2, by2 = cx + 10 + 150, cy + CAM_H - 10
        px1, py1 = bx1, by2 + 4
        px2, py2 = bx2, by2 + 10
        prog_col  = GREEN if held_frac < 1.0 else AMBER
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (40, 40, 40), -1)
        fx = px1 + int((px2 - px1) * held_frac)
        if fx > px1:
            cv2.rectangle(canvas, (px1, py1), (fx, py2), prog_col, -1)

    # ── Ký hiệu đang nhận dạng + bounding box ────────────────────────────
    if results.multi_hand_landmarks and smoothed_char:
        draw_current_char(canvas, smoothed_char, current_conf,
                          results.multi_hand_landmarks[0], CAM_W, CAM_H)

    # ── Gợi ý từ ──────────────────────────────────────────────────────────
    draw_suggestions(canvas, suggestions, selected_sug)

    # ── Text box ──────────────────────────────────────────────────────────
    tx = BORDER
    ty = BORDER + CAM_H + BORDER
    tw = CAM_W
    th = TEXT_H
    cv2.rectangle(canvas, (tx - BORDER, ty - BORDER),
                  (tx + tw + BORDER - 1, ty + th + BORDER - 1), WHITE, BORDER)
    cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), BG, -1)

    cursor  = "|" if int(time.time() * 2) % 2 == 0 else " "
    display = built_text + cursor
    max_w   = tw - PAD * 2
    while display:
        (w, _), _ = cv2.getTextSize(display, FONT, 0.9, 2)
        if w <= max_w:
            break
        display = display[1:]

    text_y = ty + (th + 20) // 2
    cv2.putText(canvas, display, (tx + PAD, text_y),
                FONT, 0.9, WHITE, 2, cv2.LINE_AA)

    # ── Hướng dẫn phím tắt ────────────────────────────────────────────────
    draw_hotkeys(canvas, show_hotkeys)

    cv2.imshow(WIN_NAME, canvas)

    # ── Xử lý phím ────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == 8:                          # Backspace — xóa ký tự cuối
        built_text     = built_text[:-1]
        last_committed = built_text[-1] if built_text else ""
        suggestions    = get_suggestions(built_text)
        selected_sug   = -1

    elif key == ord(' '):                   # Space
        if built_text and built_text[-1] != " ":
            built_text  += " "
            suggestions  = get_suggestions(built_text)
            selected_sug = -1

    elif key == ord('c'):                   # Clear all
        built_text     = ""
        last_committed = ""
        suggestions    = []
        selected_sug   = -1

    elif key == ord('g'):                   # G — toggle GUIDE panel
        show_hotkeys = not show_hotkeys

    elif key == 9:                          # Tab — chuyển sang gợi ý tiếp
        if suggestions:
            selected_sug = (selected_sug + 1) % len(suggestions)

    elif key == 13:                         # Enter — áp dụng gợi ý đang chọn
        if suggestions and selected_sug >= 0:
            built_text     = apply_suggestion(built_text, suggestions[selected_sug])
            last_committed = ""
            suggestions    = get_suggestions(built_text)
            selected_sug   = -1

cap.release()
cv2.destroyAllWindows()
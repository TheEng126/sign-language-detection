import pickle
import collections
import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# 1. Tải model và scaler
# ============================================================
model_dict = pickle.load(open('./model.p', 'rb'))
model      = model_dict['model']
scaler     = model_dict.get('scaler')
MODEL_NAME = model_dict.get('model_name', 'SVM_Model')

TOTAL_FEATURES = 63


# ============================================================
# 2. Hàm trích xuất đặc trưng (63 features - khớp với create_dataset.py)
# ============================================================
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
    return np.array(features)  # 10 features


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
    ])  # 8 features


def get_cross_finger_features(lm):
    v_index  = np.array([lm[8].x - lm[5].x, lm[8].y - lm[5].y])
    v_middle = np.array([lm[12].x - lm[9].x, lm[12].y - lm[9].y])
    v_index  = v_index  / (np.linalg.norm(v_index)  + 1e-6)
    v_middle = v_middle / (np.linalg.norm(v_middle) + 1e-6)
    dot_product   = np.dot(v_index, v_middle)
    # Tính cross product 2D thủ công (tránh DeprecationWarning của numpy 2.0)
    cross_product = v_index[0] * v_middle[1] - v_index[1] * v_middle[0]
    lateral_gap   = abs(lm[8].x - lm[12].x)
    return np.array([dot_product, cross_product, lateral_gap])  # 3 features


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


# ============================================================
# 3. Khởi tạo camera và MediaPipe
# ============================================================
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

SMOOTH_WINDOW      = 10
prediction_buffer  = collections.deque(maxlen=SMOOTH_WINDOW)

# ============================================================
# 4. Vòng lặp nhận diện realtime
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    H, W, _   = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        hand_landmarks = results.multi_hand_landmarks[0]
        features = get_enriched_features(hand_landmarks)

        if len(features) != TOTAL_FEATURES:
            prediction_buffer.clear()
        else:
            full_features = features.reshape(1, -1)
            if scaler is not None:
                full_features = scaler.transform(full_features)

            proba    = model.predict_proba(full_features)[0]
            top3_idx = np.argsort(proba)[::-1][:3]
            raw_char = model.classes_[top3_idx[0]]

            prediction_buffer.append(raw_char)
            smoothed_char = collections.Counter(prediction_buffer).most_common(1)[0][0]

            # --- UI hiển thị ---
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            x1 = max(0, int(min(x_list) * W) - 20)
            y1 = max(0, int(min(y_list) * H) - 20)
            x2 = min(W, int(max(x_list) * W) + 20)
            y2 = min(H, int(max(y_list) * H) + 20)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6
            (tw, th), _ = cv2.getTextSize(smoothed_char, font, scale, thick)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = min(H - 5, y2 + th + 10)
            cv2.putText(frame, smoothed_char, (tx, ty), font, scale, (0, 0, 0),     thick + 4, cv2.LINE_AA)
            cv2.putText(frame, smoothed_char, (tx, ty), font, scale, (255, 255, 255), thick,   cv2.LINE_AA)

            # Hiển thị xác suất Top 3
            for rank, idx in enumerate(top3_idx):
                char_i, prob_i = model.classes_[idx], proba[idx]
                y_pos   = 58 + rank * 36
                bar_len = int(180 * prob_i)
                cv2.rectangle(frame, (W-200, y_pos-18), (W-20,          y_pos+4), (50, 50, 50),  -1)
                cv2.rectangle(frame, (W-200, y_pos-18), (W-200+bar_len, y_pos+4), (0, 255, 0),   -1)
                cv2.putText(frame, f"{char_i}: {prob_i*100:.1f}%", (W-198, y_pos-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        prediction_buffer.clear()

    cv2.imshow('ASL SVM Classifier - Q to Exit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

TOTAL_FEATURES = 63  # 42 base + 10 distance + 8 angle + 3 cross-finger


# ============================================================
# Các hàm trích xuất đặc trưng bổ sung
# ============================================================

def get_distance_features(lm):
    """10 features: khoảng cách Euclidean giữa các cặp landmark quan trọng,
    normalize theo khoảng cách wrist (0) -> gốc ngón giữa (9)."""
    key_pairs = [
        (8, 12),   # đầu ngón trỏ - đầu ngón giữa  (U: gần | V: xa | R: chéo)
        (7, 11),   # khớp giữa ngón trỏ - khớp giữa ngón giữa
        (6, 10),   # khớp dưới ngón trỏ - khớp dưới ngón giữa
        (8, 4),    # đầu ngón trỏ - đầu ngón cái
        (12, 4),   # đầu ngón giữa - đầu ngón cái
        (0, 8),    # wrist - đầu ngón trỏ
        (0, 12),   # wrist - đầu ngón giữa
        (0, 4),    # wrist - đầu ngón cái
        (8, 16),   # đầu ngón trỏ - đầu ngón áp út
        (12, 16),  # đầu ngón giữa - đầu ngón áp út
    ]

    ref = np.sqrt((lm[9].x - lm[0].x)**2 + (lm[9].y - lm[0].y)**2)
    ref = max(ref, 1e-6)

    features = []
    for i, j in key_pairs:
        d = np.sqrt((lm[i].x - lm[j].x)**2 + (lm[i].y - lm[j].y)**2)
        features.append(d / ref)

    return np.array(features)  # shape (10,)


def get_angle_features(lm):
    """8 features: góc uốn tại các khớp ngón trỏ, ngón giữa và ngón cái."""
    def angle_3pts(a, b, c):
        v1 = np.array([lm[a].x - lm[b].x, lm[a].y - lm[b].y])
        v2 = np.array([lm[c].x - lm[b].x, lm[c].y - lm[b].y])
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.arccos(np.clip(cos_a, -1.0, 1.0))

    features = [
        angle_3pts(6, 7, 8),    # đốt giữa ngón trỏ
        angle_3pts(5, 6, 7),    # đốt dưới ngón trỏ
        angle_3pts(10, 11, 12), # đốt giữa ngón giữa
        angle_3pts(9, 10, 11),  # đốt dưới ngón giữa
        angle_3pts(8, 6, 10),   # góc ngón trỏ so với ngón giữa (U vs R vs V)
        angle_3pts(5, 0, 9),    # gốc ngón trỏ - cổ tay - gốc ngón giữa
        angle_3pts(2, 3, 4),    # đốt giữa ngón cái
        angle_3pts(1, 2, 3),    # đốt dưới ngón cái
    ]

    return np.array(features)  # shape (8,)


def get_cross_finger_features(lm):
    """3 features: quan hệ hướng giữa ngón trỏ và ngón giữa.
    Đây là feature mạnh nhất để phân biệt U (song song) vs R (chéo nhau)."""
    v_index  = np.array([lm[8].x - lm[5].x, lm[8].y - lm[5].y])
    v_middle = np.array([lm[12].x - lm[9].x, lm[12].y - lm[9].y])

    v_index  = v_index  / (np.linalg.norm(v_index)  + 1e-6)
    v_middle = v_middle / (np.linalg.norm(v_middle) + 1e-6)

    dot_product   = np.dot(v_index, v_middle)    # ~1 nếu song song (U), thấp hơn nếu chéo (R)
    cross_product = np.cross(v_index, v_middle)  # ~0 nếu song song, khác 0 nếu chéo (R)
    lateral_gap   = abs(lm[8].x - lm[12].x)     # khoảng cách ngang 2 đầu ngón (U: nhỏ, V: lớn)

    return np.array([dot_product, cross_product, lateral_gap])  # shape (3,)


def get_enriched_features(hand_landmarks):
    """Tổng hợp toàn bộ đặc trưng: 42 base + 10 dist + 8 angle + 3 cross = 63 features."""
    lm = hand_landmarks.landmark
    x_list = [l.x for l in lm]
    y_list = [l.y for l in lm]
    x_min, y_min = min(x_list), min(y_list)

    # 42 features gốc (tọa độ xy normalize theo bounding box)
    base = []
    for l in lm:
        base.append(l.x - x_min)
        base.append(l.y - y_min)

    dist   = get_distance_features(lm)        # 10 features
    angles = get_angle_features(lm)           # 8 features
    cross  = get_cross_finger_features(lm)    # 3 features

    return np.concatenate([base, dist, angles, cross])  # shape (63,)


# ============================================================
# Xử lý dataset
# ============================================================

data, labels = [], []

for class_label in CLASSES:
    class_dir = os.path.join(DATA_DIR, class_label)
    if not os.path.exists(class_dir):
        continue

    print(f"Đang xử lý '{class_label}'...")
    for img_name in os.listdir(class_dir):
        img = cv2.imread(os.path.join(class_dir, img_name))
        if img is None:
            continue

        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            features = get_enriched_features(results.multi_hand_landmarks[0])

            if len(features) == TOTAL_FEATURES:
                data.append(features.tolist())
                labels.append(class_label)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✓ Đã lưu {TOTAL_FEATURES} features vào data.pickle ({len(data)} mẫu)")
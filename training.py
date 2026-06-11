import pickle
import numpy as np
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

TEST_DIR       = './data_test'
DATA_PICKLE    = './data.pickle'
MODEL_PATH     = './model.p'
CLASSES        = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
TOTAL_FEATURES = 63

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


# ============================================================
# Hàm trích xuất đặc trưng (63 features)
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
    dot_product   = np.dot(v_index, v_middle)
    cross_product = v_index[0] * v_middle[1] - v_index[1] * v_middle[0]
    lateral_gap   = abs(lm[8].x - lm[12].x)
    return np.array([dot_product, cross_product, lateral_gap])


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


# ============================================================
# 1. Train model từ data.pickle → ghi đè model.p
# ============================================================
print(f"--- Bắt đầu train model từ {DATA_PICKLE} ---")

with open(DATA_PICKLE, 'rb') as f:
    data_dict = pickle.load(f)

X = np.array(data_dict['data'])
y = np.array(data_dict['labels'])
print(f"✓ Load {len(X)} mẫu | {X.shape[1]} features | {len(set(y))} lớp")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

print("Đang train SVM... (có thể mất vài phút)")
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
model.fit(X_train, y_train)

val_acc = accuracy_score(y_val, model.predict(X_val))
print(f"✓ Validation accuracy: {val_acc*100:.2f}%")

with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'model_name': 'SVM_RBF'}, f)
print(f"✓ Đã ghi đè model mới vào {MODEL_PATH}\n")


# ============================================================
# 2. Đánh giá trên tập data_test
# ============================================================
print(f"--- Đánh giá trên tập: {TEST_DIR} ---")

y_true, y_pred = [], []

for class_label in CLASSES:
    class_path = os.path.join(TEST_DIR, class_label)
    if not os.path.exists(class_path):
        continue

    for img_name in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path, img_name))
        if img is None:
            continue

        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            features = get_enriched_features(results.multi_hand_landmarks[0])
            if len(features) != TOTAL_FEATURES:
                continue
            features_scaled = scaler.transform(features.reshape(1, -1))
            y_true.append(class_label)
            y_pred.append(model.predict(features_scaled)[0])

if len(y_true) == 0:
    print("Không tìm thấy dữ liệu hợp lệ trong thư mục data_test.")
else:
    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, zero_division=0))

    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Chuẩn hóa confusion matrix theo hàng (theo nhãn thực tế) → đơn vị %
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Tỉ lệ (%)'}
    )
    plt.title('Confusion Matrix - ASL Recognition (SVM) [%]')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.tight_layout()
    plt.show()
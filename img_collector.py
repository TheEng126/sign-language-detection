import os
import cv2
import time
import numpy as np
import mediapipe as mp

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

CLASSES     = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
DATASET_SIZE    = 200
CAPTURE_INTERVAL = 0.4
BLUR_THRESHOLD   = 80.0

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

def is_sharp(frame, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    var  = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var >= threshold, var

def has_hand(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    return result.multi_hand_landmarks is not None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()
for class_label in CLASSES:
    class_dir = os.path.join(DATA_DIR, class_label)
    os.makedirs(class_dir, exist_ok=True)

    existing = len(os.listdir(class_dir))
    if existing >= DATASET_SIZE:
        print(f"[BỎ QUA] '{class_label}' đã đủ {existing} ảnh.")
        continue

    print(f'\n=== Ký tự: {class_label} ({existing}/{DATASET_SIZE}) ===  Nhấn S để bắt đầu, Q để thoát')

    # ---------- Màn hình chờ ----------
    while True:
        ret, raw = cap.read()
        if not ret:
            break

        raw     = cv2.flip(raw, 1)
        preview = raw.copy()          # Vẽ UI lên preview, KHÔNG đụng raw

        cv2.putText(preview, f'Ky hieu: {class_label}', (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, 'Nhan S de bat dau  |  Q de thoat', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Thu thap du lieu - ASL', preview)

        key = cv2.waitKey(25) & 0xFF
        if key in (ord('s'), ord('S')):
            break
        if key in (ord('q'), ord('Q')):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # ---------- Vòng chụp ảnh ----------
    counter          = existing
    skipped          = 0
    last_capture     = time.time()

    while counter < DATASET_SIZE:
        ret, raw = cap.read()
        if not ret:
            break

        raw     = cv2.flip(raw, 1)    # ← ảnh GỐC, chưa vẽ gì
        preview = raw.copy()          # ← bản sao để vẽ UI

        now          = time.time()
        sharp, var   = is_sharp(raw)

        # --- Vẽ UI lên preview ---
        bar_fill  = int((counter / DATASET_SIZE) * (preview.shape[1] - 60))
        cv2.rectangle(preview, (30, preview.shape[0]-25),
                      (preview.shape[1]-30, preview.shape[0]-8), (50,50,50), -1)
        cv2.rectangle(preview, (30, preview.shape[0]-25),
                      (30 + bar_fill, preview.shape[0]-8), (0,220,0), -1)

        status_col  = (0, 255, 0) if sharp else (0, 0, 220)
        status_txt  = f'Sharp:{var:.0f}' if sharp else f'Nhoe:{var:.0f} - Giu tay co dinh!'
        cv2.putText(preview, f'{class_label}: {counter}/{DATASET_SIZE}',
                    (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3, cv2.LINE_AA)
        cv2.putText(preview, status_txt,
                    (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2, cv2.LINE_AA)
        cv2.putText(preview, f'Bo qua: {skipped}',
                    (30, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2, cv2.LINE_AA)

        cv2.imshow('Thu thap du lieu - ASL', preview)   # Chỉ hiện preview

        # --- Chụp: lưu raw (không có chữ) ---
        if now - last_capture >= CAPTURE_INTERVAL:
            if sharp and has_hand(raw):
                cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), raw)  # ← lưu raw
                counter += 1
            else:
                skipped += 1
            last_capture = now

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            print("Đã thoát sớm.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f'  --> Lưu: {counter} ảnh sạch | Bỏ qua: {skipped}')

cap.release()
cv2.destroyAllWindows()
print('\n=== Hoàn thành thu thập 26 ký tự A-Z ===')
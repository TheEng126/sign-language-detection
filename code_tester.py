import os
import cv2
import time
import mediapipe as mp

DATA_DIR = './data_test'

# ============================================================
# Chỉ định ký tự cần thu thêm và số ảnh muốn thêm
# ============================================================
EXTRA_CLASSES  = ['W', 'X']
EXTRA_PER_CLASS = 30              # số ảnh thu thêm mỗi ký tự

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

for class_label in EXTRA_CLASSES:
    class_dir = os.path.join(DATA_DIR, class_label)
    os.makedirs(class_dir, exist_ok=True)

    # Đếm ảnh hiện có để đặt tên file tiếp nối, không ghi đè
    existing_files = os.listdir(class_dir)
    existing_count = len(existing_files)
    target_count   = existing_count + EXTRA_PER_CLASS

    print(f"\n=== '{class_label}': hiện có {existing_count} ảnh, thu thêm {EXTRA_PER_CLASS} → tổng {target_count} ===")
    print('Nhấn S để bắt đầu  |  Q để thoát')
    print('>> Hãy thực hiện ký hiệu theo góc bị nhận sai (vd: xoay 180°)')

    # ---------- Màn hình chờ ----------
    while True:
        ret, raw = cap.read()
        if not ret:
            break
        raw     = cv2.flip(raw, 1)
        preview = raw.copy()

        cv2.putText(preview, f'Thu them: {class_label}', (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(preview, f'Hien co: {existing_count} anh | Se thu them: {EXTRA_PER_CLASS}', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, 'Nhan S de bat dau  |  Q de thoat', (30, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.imshow('Thu them data - ASL', preview)

        key = cv2.waitKey(25) & 0xFF
        if key in (ord('s'), ord('S')):
            break
        if key in (ord('q'), ord('Q')):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # ---------- Vòng chụp ảnh ----------
    # Đặt tên file bắt đầu từ existing_count để không ghi đè ảnh cũ
    counter      = existing_count
    skipped      = 0
    last_capture = time.time()

    while counter < target_count:
        ret, raw = cap.read()
        if not ret:
            break

        raw     = cv2.flip(raw, 1)
        preview = raw.copy()
        now     = time.time()
        sharp, var = is_sharp(raw)

        # UI
        added     = counter - existing_count
        bar_fill  = int((added / EXTRA_PER_CLASS) * (preview.shape[1] - 60))
        cv2.rectangle(preview, (30, preview.shape[0]-25),
                      (preview.shape[1]-30, preview.shape[0]-8), (50,50,50), -1)
        cv2.rectangle(preview, (30, preview.shape[0]-25),
                      (30 + bar_fill, preview.shape[0]-8), (0,220,255), -1)

        status_col = (0, 255, 0) if sharp else (0, 0, 220)
        status_txt = f'Sharp:{var:.0f}' if sharp else f'Nhoe:{var:.0f} - Giu tay!'
        cv2.putText(preview, f'{class_label}: +{added}/{EXTRA_PER_CLASS}',
                    (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3, cv2.LINE_AA)
        cv2.putText(preview, status_txt,
                    (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2, cv2.LINE_AA)
        cv2.putText(preview, f'Bo qua: {skipped}',
                    (30, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2, cv2.LINE_AA)

        cv2.imshow('Thu them data - ASL', preview)

        if now - last_capture >= CAPTURE_INTERVAL:
            if sharp and has_hand(raw):
                # Tên file tiếp nối từ existing_count, không đụng file cũ
                img_path = os.path.join(class_dir, f'{counter}.jpg')
                cv2.imwrite(img_path, raw)
                counter += 1
            else:
                skipped += 1
            last_capture = now

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            print("Đã thoát sớm.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    added_total = counter - existing_count
    print(f"  --> Đã thêm: {added_total} ảnh | Bỏ qua: {skipped} | Tổng '{class_label}': {counter} ảnh")

cap.release()
cv2.destroyAllWindows()
print('\n=== Hoàn thành thu thêm data ===')
print('Chạy lại create_dataset.py và train_classifier.py để cập nhật model.')
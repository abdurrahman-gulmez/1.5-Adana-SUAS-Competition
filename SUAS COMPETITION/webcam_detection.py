import numpy as np
import os
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Model ve URL Bilgileri
MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'  # Doğru model adı
MODEL_DATE = '20200711'  # Modelin yayın tarihi
MODEL_TAR = f'{MODEL_NAME}.tar.gz'
BASE_URL = f'http://download.tensorflow.org/models/object_detection/tf2/{MODEL_DATE}/'  # Doğru URL

# Dosya Yolları
MODELS_DIR = os.path.join('data', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
PATH_TO_CKPT = os.path.join(MODELS_DIR, MODEL_NAME, 'saved_model')
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Modeli İndirme (Yeni URL ile)
if not os.path.exists(PATH_TO_CKPT):
    tf.keras.utils.get_file(
        fname=MODEL_TAR,
        origin=BASE_URL + MODEL_TAR,
        cache_dir=MODELS_DIR,
        cache_subdir='',
        extract=True
    )
    os.remove(os.path.join(MODELS_DIR, MODEL_TAR))  # .tar.gz dosyasını sil

# Etiket Haritasını Yükle
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Modeli Yükle (TF2 SavedModel)
print("Loading model...")
model = tf.saved_model.load(PATH_TO_CKPT)
infer = model.signatures['serving_default']  # Çıkarım imzası

# Webcam Bağlantısı
cap = cv2.VideoCapture(0)

# Ana Döngü
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Girdiyi Hazırla (Model 320x320 bekler)
    frame = cv2.flip(frame, 1)
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    input_tensor = tf.image.resize(input_tensor, [320, 320])
    input_tensor = tf.cast(input_tensor, tf.uint8)  # Kritik dönüşüm

    # Çıkarım Yap
    detections = infer(input_tensor)

    # Sonuçları İşle
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    # Görselleştirme (Boyutları orijinale uyarla)
    frame_resized = cv2.resize(frame, (320, 320))
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame_resized,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.5
    )
    frame = cv2.resize(frame_resized, (800, 600))

    cv2.imshow('TF2 Object Detection - SSD MobileNet V2', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
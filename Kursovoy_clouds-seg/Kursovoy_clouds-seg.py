# Подключение библиотек
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import urllib.request
import tarfile
import gzip
import shutil
import matplotlib.pyplot as plt
import segmentation_models as sm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gc
tf.keras.backend.clear_session()
gc.collect()

# ================================
# ЗАГРУЗКА И ПРОВЕРКА ДАННЫХ
# ================================
def download_file(url, filename):
    """Скачивает файл если он отсутствует"""
    if not os.path.exists(filename):
        print(f"Скачивание {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ {filename} загружен")
    else:
        print(f"✅ {filename} уже существует")

def extract_tar_gz(filename, extract_path):
    """Распаковывает tar.gz архив"""
    if not os.path.exists(extract_path):
        print(f"Распаковка {filename}...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        print(f"✅ {filename} распакован в {extract_path}")
    else:
        print(f"✅ {extract_path} уже существует")

def extract_gz(filename):
    """Распаковывает gz архив"""
    output_filename = filename.replace('.gz', '')
    if not os.path.exists(output_filename):
        print(f"Распаковка {filename}...")
        with gzip.open(filename, 'rb') as f_in:
            with open(output_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✅ {filename} распакован в {output_filename}")
    else:
        print(f"✅ {output_filename} уже существует")

def check_required_files():
    """Проверяет наличие всех необходимых файлов"""
    required_files = {
        'train.csv': 'https://video.ittensive.com/machine-learning/clouds/train.csv.gz',
        'train_images_small': 'https://video.ittensive.com/machine-learning/clouds/train_images_small.tar.gz',
        'test_images_small': 'https://video.ittensive.com/machine-learning/clouds/test_images_small.tar.gz',
        'sample_submission.csv': 'https://video.ittensive.com/machine-learning/clouds/sample_submission.csv.gz'
    }
    
    print("=" * 50)
    print("ПРОВЕРКА И ЗАГРУЗКА ДАННЫХ")
    print("=" * 50)
    
    # Скачиваем и распаковываем файлы
    for file_key, url in required_files.items():
        if file_key.endswith('_images_small'):
            # Для архивов с изображениями
            archive_name = f"{file_key}.tar.gz"
            download_file(url, archive_name)
            extract_tar_gz(archive_name, file_key)
        elif file_key.endswith('.csv'):
            # Для CSV файлов
            gz_name = f"{file_key}.gz"
            download_file(url, gz_name)
            extract_gz(gz_name)
    
    # Проверяем наличие всех файлов
    all_exists = True
    for file_key in required_files.keys():
        if file_key.endswith('_images_small'):
            if not os.path.exists(file_key):
                print(f"❌ Отсутствует: {file_key}")
                all_exists = False
            else:
                # Проверяем что в директории есть файлы
                files = os.listdir(file_key)
                print(f"✅ {file_key}: {len(files)} файлов")
        else:
            if not os.path.exists(file_key):
                print(f"❌ Отсутствует: {file_key}")
                all_exists = False
            else:
                file_size = os.path.getsize(file_key) / (1024 * 1024)  # в МБ
                print(f"✅ {file_key}: {file_size:.1f} Мб")
    
    if all_exists:
        print("🎉 Все файлы готовы к работе!")
    else:
        print("⚠️ Некоторые файлы отсутствуют!")
    
    return all_exists

# Проверяем и загружаем данные
if not check_required_files():
    print("❌ Не все файлы загружены. Прерывание выполнения.")
    exit(1)

# ================================
# ПАРАМЕТРЫ
# ================================
BATCH_SIZE = 1
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3
TRAIN_DIR = "train_images_small"
TEST_DIR = "test_images_small"

sm.set_framework('tf.keras')

# ================================
# RLE утилиты
# ================================
def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs) if len(runs) > 0 else ''

def rle_decode(mask_rle, shape=(350, 525)):
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# ================================
# Загрузка данных
# ================================
def load_y(df, target_shape=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    masks = []
    for rle in df["EncodedPixels"]:
        mask = rle_decode(rle, shape=(350, 525))
        mask_img = tf.keras.utils.array_to_img(mask[:, :, np.newaxis])
        mask_resized = mask_img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        mask_array = tf.keras.utils.img_to_array(mask_resized).squeeze(-1)
        masks.append((mask_array > 0.5).astype(np.float32))
    return np.array(masks)[:, :, :, np.newaxis]

def load_x(df, data_dir, target_shape=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    imgs = np.empty((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), dtype=np.float32)
    for i, fname in enumerate(df["Image"]):
        img = tf.keras.utils.load_img(
            os.path.join(data_dir, fname),
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
        )
        imgs[i] = tf.keras.utils.img_to_array(img)
    return imgs

# ================================
# Dice
# ================================
def dice_coef_np(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * y_pred_bin)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred_bin) + smooth)

# ================================
# Загрузка train данных
# ================================
print("Загрузка меток...")
data = pd.read_csv('train.csv')
data["Image"] = data["Image_Label"].str.split("_", expand=True)[0]
data["Label"] = data["Image_Label"].str.split("_", expand=True)[1]
data_fish = data[data["Label"] == "Fish"].copy()
data_fish.drop(columns=["Image_Label", "Label"], inplace=True)
del data

train_val, _ = train_test_split(data_fish, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val, test_size=0.2, random_state=42)
del train_val, data_fish

print(f"Train: {len(train_df)}, Val: {len(val_df)}")

X_train = load_x(train_df, TRAIN_DIR)
y_train = load_y(train_df)
X_val = load_x(val_df, TRAIN_DIR)
y_val = load_y(val_df)

# ================================
# Аугментация
# ================================
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

# ================================
# Обучение модели
# ================================
def train_model(model_name, backbone, X_train, y_train, X_val, y_val, epochs=12):
    print(f"\nОбучение: {model_name} + {backbone}")
    preprocess = sm.get_preprocessing(backbone)
    X_train_p = preprocess(X_train)
    X_val_p = preprocess(X_val)

    if model_name == 'FPN':
        model = sm.FPN(backbone, encoder_weights='imagenet', classes=1, activation='sigmoid')
    elif model_name == 'Unet':
        model = sm.Unet(backbone, encoder_weights='imagenet', classes=1, activation='sigmoid')
    else:
        raise ValueError("Поддерживаемые модели: FPN, Unet")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=sm.losses.bce_dice_loss,
        metrics=[sm.metrics.iou_score, 'binary_accuracy']
    )

    callbacks = [
        ModelCheckpoint(f"{model_name}_{backbone}_best.h5", monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    model.fit(
        datagen.flow(X_train_p, y_train, batch_size=BATCH_SIZE, seed=42),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_val_p, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return model, preprocess

# ================================
# Обучаем две модели
# ================================
model1, preprocess1 = train_model('FPN', 'mobilenetv2', X_train, y_train, X_val, y_val, epochs=20)
model2, preprocess2 = train_model('Unet', 'resnet50', X_train, y_train, X_val, y_val, epochs=20)

# ================================
# Оценка по Dice и подбор порога
# ================================
print("\nВычисление предсказаний на валидации...")
X_val_p1 = preprocess1(X_val)
X_val_p2 = preprocess2(X_val)

preds1 = model1.predict(X_val_p1, batch_size=BATCH_SIZE, verbose=0)
preds2 = model2.predict(X_val_p2, batch_size=BATCH_SIZE, verbose=0)
preds_ens = (preds1 + preds2) / 2.0

# Оценка отдельных моделей
dice1 = [dice_coef_np(y_val[i, :, :, 0], preds1[i, :, :, 0], threshold=0.4) for i in range(len(y_val))]
dice2 = [dice_coef_np(y_val[i, :, :, 0], preds2[i, :, :, 0], threshold=0.4) for i in range(len(y_val))]
print(f"\nСредний Dice (порог=0.4):")
print(f"  FPN (MobileNetV2): {np.mean(dice1):.4f}")
print(f"  Unet (ResNet50):   {np.mean(dice2):.4f}")

# Подбор лучшего порога для ансамбля
best_thresh, best_dice = 0.3, 0.0
thresholds = np.arange(0.3, 0.61, 0.025)
print("\nПоиск лучшего порога по Dice ансамбля...")
for th in thresholds:
    dice_scores = [
        dice_coef_np(y_val[i, :, :, 0], preds_ens[i, :, :, 0], threshold=th)
        for i in range(len(y_val))
    ]
    avg_dice = np.mean(dice_scores)
    if avg_dice > best_dice:
        best_dice, best_thresh = avg_dice, th
    print(f"  Порог {th:.3f} → Dice = {avg_dice:.4f}")

print(f"\n✅ Лучший порог: {best_thresh:.3f}, Dice = {best_dice:.4f}")
THRESHOLD = best_thresh

# ================================
# Визуализация: оригинал, маска и предсказание (БЕЗ GUI)
# ================================
print("\nСохранение примера сегментации...")

idx = 0
orig_img = X_val[idx].astype(np.uint8)
true_mask = y_val[idx, :, :, 0].astype(np.uint8)
pred_mask = preds_ens[idx, :, :, 0]
pred_mask_binary = (pred_mask > THRESHOLD).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(orig_img)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(orig_img)
axes[1].imshow(true_mask, cmap='Reds', alpha=0.5)
axes[1].set_title("Ground Truth Mask")
axes[1].axis("off")

axes[2].imshow(orig_img)
axes[2].imshow(pred_mask_binary, cmap='Reds', alpha=0.5)
axes[2].set_title(f"Ensemble Prediction (Threshold={THRESHOLD:.3f})")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("segmentation_example.png", dpi=150, bbox_inches='tight')
print("✅ Пример сохранён: segmentation_example.png")

# Оценка ансамбля с лучшим порогом
dice_ens_final = [
    dice_coef_np(y_val[i, :, :, 0], preds_ens[i, :, :, 0], threshold=THRESHOLD)
    for i in range(len(y_val))
]
print(f"✅ Ансамбль (финальный Dice): {np.mean(dice_ens_final):.4f}")

# ================================
# ПРЕДСКАЗАНИЕ НА TEST ДАННЫХ - ИСПРАВЛЕННАЯ ВЕРСИЯ
# ================================
print("\n📊 ПРЕДСКАЗАНИЕ НА TEST ДАННЫХ...")

submission = pd.read_csv('sample_submission.csv')
submission["Image"] = submission["Image_Label"].str.split("_", expand=True)[0]
submission["Label"] = submission["Image_Label"].str.split("_", expand=True)[1]

test_images = submission[submission["Label"] == "Fish"]["Image"].unique()
print(f"Всего тестовых изображений для Fish: {len(test_images)}")

predictions = []
for i, img_name in enumerate(test_images):
    if i % 50 == 0:  # Реже выводим прогресс
        print(f"Обработано {i}/{len(test_images)} изображений...")
    
    img_path = os.path.join(TEST_DIR, img_name)
    img = tf.keras.utils.load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    pred1 = model1.predict(preprocess1(img_batch), verbose=0)[0, :, :, 0]
    pred2 = model2.predict(preprocess2(img_batch), verbose=0)[0, :, :, 0]
    pred_avg = (pred1 + pred2) / 2.0
    
    # Масштабируем предсказание обратно к оригинальному размеру (350, 525)
    pred_resized = tf.image.resize(
        pred_avg[np.newaxis, :, :, np.newaxis], 
        (350, 525), 
        method='bilinear'
    ).numpy()[0, :, :, 0]
    
    pred_bin = (pred_resized > THRESHOLD).astype(np.uint8)
    rle = rle_encode(pred_bin)
    predictions.append((img_name + "_Fish", rle))

# ================================
# ФОРМИРОВАНИЕ SUBMISSION - ИСПРАВЛЕННАЯ ВЕРСИЯ
# ================================
print("\nФормирование submission файла...")

# Создаем DataFrame для предсказаний Fish
submission_fish = pd.DataFrame(predictions, columns=["Image_Label", "EncodedPixels"])

# Объединяем с оригинальным submission
final_submission = submission[["Image_Label"]].copy()
final_submission = final_submission.merge(submission_fish, on="Image_Label", how="left")
final_submission["EncodedPixels"].fillna("", inplace=True)

# Проверяем результат
fish_predictions = final_submission[final_submission["Image_Label"].str.endswith("_Fish")]
non_empty = fish_predictions[fish_predictions["EncodedPixels"] != ""]

print(f"Всего строк в submission: {len(final_submission)}")
print(f"Предсказания для Fish: {len(fish_predictions)}")
print(f"Непустые предсказания Fish: {len(non_empty)}")

final_submission.to_csv("submission.csv", index=False)
print(f"\n✅ Submission сохранён: submission.csv")

print("\nПример предсказаний:")
print(non_empty.head(10))

print("\nСтатистика по классам:")
for label in ["Fish", "Flower", "Gravel", "Sugar"]:
    label_predictions = final_submission[final_submission["Image_Label"].str.endswith(f"_{label}")]
    non_empty_count = len(label_predictions[label_predictions["EncodedPixels"] != ""])
    print(f"  {label}: {non_empty_count}/{len(label_predictions)} непустых")

# ================================
# ФИНАЛЬНЫЙ ОТЧЕТ
# ================================
print("\n" + "=" * 50)
print("ФИНАЛЬНЫЙ ОТЧЕТ")
print("=" * 50)

print("📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
print(f"🎯 Оптимальный порог: {THRESHOLD:.3f}")
print(f"📈 Dice на валидации: {best_dice:.4f}")

print(f"\n📁 СОЗДАННЫЕ ФАЙЛЫ:")
print("  - submission.csv - финальный submission файл")
print("  - segmentation_example.png - пример сегментации")
print("  - fpn_mobilenetv2_best.h5 - веса FPN модели")
print("  - unet_resnet50_best.h5 - веса U-Net модели")

print(f"\n📊 СТАТИСТИКА SUBMISSION:")
print(f"  Всего строк: {len(final_submission)}")
print(f"  Изображений: {len(test_images)}")
print(f"  Непустых предсказаний Fish: {len(non_empty)}")

print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
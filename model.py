import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# データセットのパス
dir_path = 'dataset/Train'
dir_path_test = 'dataset/Test'

# カテゴリとラベルのマッピング
categories = {
    'freshapples': 0,
    'freshbanana': 1,
    'freshoranges': 2,
    'rottenapples': 3,
    'rottenbanana': 4,
    'rottenoranges': 5
}

# データ読み込み関数
def load_data(data_dir, categories):
    X, Y = [], []
    for category, label in categories.items():
        path = os.path.join(data_dir, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                Y.append(label)
    return np.array(X), np.array(Y)

# データの読み込み
X_train, Y_train = load_data(dir_path, categories)
X_val, Y_val = load_data(dir_path_test, categories)

# データの正規化
X_train = X_train / 255.0
X_val = X_val / 255.0

# モデルの構築
mobilenetv2_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
mobilenetv2_model.trainable = False

model = Sequential([
    mobilenetv2_model,
    BatchNormalization(),
    SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
    SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(6, activation='softmax')
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# コールバックの設定
lr_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1, mode='max', min_lr=0.00002, cooldown=2)
check_point = ModelCheckpoint(filepath='modelcheckpt.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# モデルの学習
history = model.fit(X_train, Y_train, batch_size=32, validation_data=(X_val, Y_val), epochs=10, callbacks=[check_point, lr_rate])

# 学習結果のプロット
plt.figure(figsize=(20, 12))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()

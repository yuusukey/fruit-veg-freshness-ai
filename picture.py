import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog
from PIL import Image as PILImage

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
label_map = {v: k for k, v in categories.items()}

# データ読み込み関数
def load_data(data_dir, categories):
    X, Y = [], []
    for category, label in categories.items():
        path = os.path.join(data_dir, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                Y.append(label)
    return np.array(X), np.array(Y)

# データの読み込み
X_train, Y_train = load_data(dir_path, categories)
X_val, Y_val = load_data(dir_path_test, categories)

# データの前処理
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)

# モデルの構築
mobilenetv2_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
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
check_point = ModelCheckpoint(filepath='modelcheckpt.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

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

# 学習済みモデルの保存
model.save('fruit_freshness_model.keras')

# # 画像を挿入して鮮度を判別する機能
# def predict_freshness(model_path):
#     # モデルの読み込み
#     model = load_model(model_path)

#     # ファイルダイアログを開いて画像を選択
#     root = Tk()
#     root.withdraw()  # メインウィンドウを非表示にする
#     file_path = filedialog.askopenfilename(title='画像を選択してください', filetypes=[('Image Files', '*.jpg;*.jpeg;*.png')])
#     if not file_path:
#         print("画像が選択されませんでした。")
#         return

#     # 画像の読み込みと前処理
#     img = PILImage.open(file_path)
#     img = img.resize((224, 224))
#     img_array = np.array(img)
#     if img_array.shape[-1] == 4:
#         img_array = img_array[..., :3]  # アルファチャネルを削除
#     img_array = preprocess_input(img_array)
#     img_array = np.expand_dims(img_array, axis=0)

#     # 予測
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions, axis=1)[0]
#     class_label = label_map[predicted_class]
#     confidence = predictions[0][predicted_class] * 100

#     print(f"予測結果: {class_label} ({confidence:.2f}%)")

# # 予測関数の実行
# predict_freshness('fruit_freshness_model.keras')

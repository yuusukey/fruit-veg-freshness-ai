import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image as PILImage
from tkinter import Tk, filedialog

# ラベルマップの定義（仮の例）
label_map = {
    0: "新鮮なりんご",
    1: "新鮮なバナナ",
    2: "新鮮なオレンジ",
    3: "腐敗したりんご",
    4: "腐敗したバナナ",
    5: "腐敗したオレンジ"
}

# 画像を挿入して鮮度を判別する機能
def predict_freshness(model_path):
    # モデルの読み込み
    model = load_model(model_path)

    # ファイルダイアログを開いて画像を選択
    root = Tk()
    root.withdraw()  # メインウィンドウを非表示にする
    file_path = filedialog.askopenfilename(title='画像を選択してください', filetypes=[('Image Files', '*.jpg;*.jpeg;*.png')])
    if not file_path:
        print("画像が選択されませんでした。")
        return

    # 画像の読み込みと前処理
    img = PILImage.open(file_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # アルファチャネルを削除
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # 予測
    predictions = model.predict(img_array)
    print(f"予測結果: {predictions}")
    predicted_class = np.argmax(predictions, axis=1)[0]
    print(f"予測クラス: {predicted_class}")
    class_label = label_map.get(predicted_class, "不明")
    confidence = predictions[0][predicted_class] * 100

    print(f"予測結果: {class_label} ({confidence:.2f}%)")

def main():
    # モデルのパス
    model_path = 'fruit_freshness_model.keras'
    
    # 予測関数の実行
    predict_freshness(model_path)

if __name__ == "__main__":
    # 予測関数の実行
    main()

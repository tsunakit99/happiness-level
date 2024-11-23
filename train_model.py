# train_model.py

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 自作モジュールのインポート
from utils.data_preprocessing import load_data
from utils.model import create_model

# データディレクトリの設定
data_dir = Path(__file__).resolve().parent / "data"
dataset_dir = data_dir / "dataset_images"

# 感情カテゴリと対応する幸せ度合いのラベル
emotion_labels = {
    'happy': 90,
    'neutral': 50,
    'sad': 20,
    'angry': 10,
    'disgust': 10,
    'fear': 10,
    'surprise': 70  # 必要に応じて調整
}

# データの読み込み
images, labels = load_data(dataset_dir, emotion_labels, image_size=(48, 48))

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# データ増強の設定
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# データ増強のために訓練データに適合
datagen.fit(X_train)

# モデルの構築
input_shape = (48, 48, 1)
model = create_model(input_shape)

# エポック数とバッチサイズの設定
epochs = 50
batch_size = 64

# モデルの訓練
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

# モデルの保存
model.save(str(Path(__file__).resolve().parent / 'models' / 'happiness_model.h5'))

# テストデータでの評価
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_loss}")
print(f"Test MAE: {test_mae}")

# 学習曲線のプロット
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig("train.png")

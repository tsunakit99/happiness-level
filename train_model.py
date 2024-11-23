import os
from pathlib import Path

import cv2
import dlib
import numpy as np
import pandas as pd  # 追加
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# データディレクトリの設定
data_dir = Path(__file__).resolve().parent / "data"
dataset_dir = data_dir / "dataset_images"

# モデルディレクトリの設定
models_dir = Path(__file__).resolve().parent / "models"
shape_predictor_path = models_dir / "shape_predictor_68_face_landmarks.dat"

# Dlibの初期化
predictor = dlib.shape_predictor(str(shape_predictor_path))

# 特徴量とラベルを保存するリスト
features = []
labels = []

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

# データセットの読み込みと特徴量抽出
for dataset_type in ['train', 'test']:
    dataset_path = dataset_dir / dataset_type
    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        continue

    for emotion, happiness_score in emotion_labels.items():
        emotion_dir = dataset_path / emotion
        if not emotion_dir.exists():
            print(f"Emotion directory not found: {emotion_dir}")
            continue

        processed_images_count = 0

        for img_file in emotion_dir.glob("*.*"):
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 顔検出をスキップ
            height, width = gray.shape
            rect = dlib.rectangle(0, 0, width, height)

            try:
                landmarks = predictor(gray, rect)
            except Exception as e:
                print(f"Failed to detect landmarks in image: {img_file}, Error: {e}")
                continue

            # ランドマークを配列に変換
            points = np.array([[p.x, p.y] for p in landmarks.parts()])

            # 顔のサイズを取得（rectを使用）
            face_width = rect.right() - rect.left()
            face_height = rect.bottom() - rect.top()

            # 特徴量の計算
            try:
                # 口の特徴量
                left_mouth_corner = points[48]
                right_mouth_corner = points[54]
                upper_lip = points[51]
                lower_lip = points[57]

                mouth_angle = np.arctan2(
                    right_mouth_corner[1] - left_mouth_corner[1],
                    right_mouth_corner[0] - left_mouth_corner[0]
                ) * (180 / np.pi)

                mouth_height = lower_lip[1] - upper_lip[1]
                normalized_mouth_height = mouth_height / face_height

                mouth_width = right_mouth_corner[0] - left_mouth_corner[0]
                normalized_mouth_width = mouth_width / face_width

                mouth_ratio = normalized_mouth_height / normalized_mouth_width

                # 目の特徴量
                left_eye = points[42:48]
                right_eye = points[36:42]

                left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5])
                right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5])
                normalized_eye_height = (left_eye_height + right_eye_height) / (2 * face_height)

                # 眉間の特徴量
                brow_center = points[21:23]
                brow_distance = np.linalg.norm(brow_center[0] - brow_center[1])
                normalized_brow_distance = brow_distance / face_width

                # 特徴量を配列にまとめる
                feature_vector = np.array([
                    mouth_angle,
                    normalized_mouth_height,
                    normalized_mouth_width,
                    mouth_ratio,
                    normalized_eye_height,
                    normalized_brow_distance
                ])

                features.append(feature_vector)
                labels.append(happiness_score)
                processed_images_count += 1
            except IndexError as e:
                print(f"Failed to compute features in image: {img_file}, Error: {e}")
                continue

        print(f"Processed {processed_images_count} images for emotion '{emotion}' in '{dataset_type}' dataset.")

# 特徴量とラベルをNumPy配列に変換
features = np.array(features)
labels = np.array(labels)

# 特徴量が存在するか確認
if len(features) == 0:
    print("No features were extracted. Exiting.")
    exit()

# 特徴量の標準化
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
features_standardized = (features - mean) / std

# 平均値と標準偏差を保存
np.save(str(data_dir / 'feature_mean.npy'), mean)
np.save(str(data_dir / 'feature_std.npy'), std)

# 特徴量とラベルを保存
np.save(str(data_dir / 'features.npy'), features_standardized)
np.save(str(data_dir / 'labels.npy'), labels)

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(features_standardized, labels, test_size=0.2, random_state=42)

# DMatrixに変換
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# モデルのパラメータ
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# クロスバリデーションで最適なブーストラウンド数を決定
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    early_stopping_rounds=50,
    metrics="rmse",
    seed=42
)

# cv_results の型に応じて処理
if isinstance(cv_results, dict):
    best_num_boost_round = len(cv_results['train-rmse-mean'])
elif isinstance(cv_results, pd.DataFrame):
    best_num_boost_round = cv_results.shape[0]
else:
    raise TypeError(f"Unexpected type for cv_results: {type(cv_results)}")

# モデルの訓練
model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round)

# モデルの評価
y_pred = model.predict(dtest)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# モデルの保存
model.save_model(str(models_dir / 'happiness_model.json'))

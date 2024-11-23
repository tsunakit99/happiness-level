import json
from pathlib import Path
from wsgiref.simple_server import make_server

import cv2
import dlib
import falcon
import numpy as np
import xgboost as xgb
from string2image import string2image


class HappinessMeterResource:
    def __init__(self):
        # Dlibの顔検出器とランドマーク検出器を初期化
        self.detector = dlib.get_frontal_face_detector()

        # モデルのパスを解決
        model_path = Path(__file__).resolve().parent.parent / "models/shape_predictor_68_face_landmarks.dat"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.predictor = dlib.shape_predictor(str(model_path))

        # 機械学習モデルと標準化パラメータをロード
        self.model = xgb.Booster()
        model_file = Path(__file__).resolve().parent.parent / "models/happiness_model.json"
        if not model_file.exists():
            raise FileNotFoundError(f"Machine learning model not found: {model_file}")
        self.model.load_model(str(model_file))

        mean_file = Path(__file__).resolve().parent.parent / "data/feature_mean.npy"
        std_file = Path(__file__).resolve().parent.parent / "data/feature_std.npy"
        if not mean_file.exists() or not std_file.exists():
            raise FileNotFoundError("Feature mean and std files not found.")

        self.mean = np.load(str(mean_file))
        self.std = np.load(str(std_file))

    def on_post(self, req, resp):
        # POSTされたデータを受け取る
        params = req.media
        img_as_text = params['imagedata']

        # Base64エンコードされた画像をデコード
        img = string2image(img_as_text, False)

        # 幸せ度合いを計算
        happiness_score, processed_image = self.calculate_happiness(img)

        # 処理済み画像を保存
        output_path = self.save_processed_image(processed_image)

        # JSON応答を作成
        response_data = {
            "happiness_score": happiness_score,
            "processed_image_path": output_path
        }

        resp.text = json.dumps(response_data)

    def calculate_happiness(self, img):
        """
        幸せ度合いを計算するメソッド。
        :param img: 入力画像
        :return: 幸せ度合いスコア, 処理済み画像
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            return 0, img  # 顔が検出されなかった場合、スコアは0

        # 最初に検出された顔を処理
        face = faces[0]
        landmarks = self.predictor(gray, face)

        # ランドマークを配列に変換
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # 顔のサイズを取得
        face_width = face.right() - face.left()
        face_height = face.bottom() - face.top()

        # 特徴量の計算
        left_mouth_corner = points[48]  # 口の左端
        right_mouth_corner = points[54]  # 口の右端
        upper_lip = points[51]  # 上唇の中央
        lower_lip = points[57]  # 下唇の中央

        # 口角の角度を計算
        mouth_angle = np.arctan2(
            right_mouth_corner[1] - left_mouth_corner[1],
            right_mouth_corner[0] - left_mouth_corner[0]
        ) * (180 / np.pi)

        # 口の高さと幅を計算し、顔の高さで正規化
        mouth_height = lower_lip[1] - upper_lip[1]
        normalized_mouth_height = mouth_height / face_height

        mouth_width = right_mouth_corner[0] - left_mouth_corner[0]
        normalized_mouth_width = mouth_width / face_width

        # 口の高さと幅の比率
        mouth_ratio = normalized_mouth_height / normalized_mouth_width

        # 目の開き具合を計算
        left_eye = points[42:48]  # 左目
        right_eye = points[36:42]  # 右目

        left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5])
        right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5])
        normalized_eye_height = (left_eye_height + right_eye_height) / (2 * face_height)

        # 眉間の距離を計算し、顔の幅で正規化
        brow_center = points[21:23]  # 眉間
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

        # 特徴量の標準化
        feature_vector_standardized = (feature_vector - self.mean) / self.std

        # 機械学習モデルで予測
        dmatrix = xgb.DMatrix([feature_vector_standardized])
        happiness_score = self.model.predict(dmatrix)[0]

        # スコアの範囲を0から100に制限
        happiness_score = float(np.clip(happiness_score, 0, 100))

        # デバッグ情報の出力
        print(f"Debug Info:")
        print(f"Features: {feature_vector}")
        print(f"Standardized Features: {feature_vector_standardized}")
        print(f"Happiness Score: {happiness_score}")

        # ランドマークを描画
        for (x, y) in points:
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        return happiness_score, img

    def save_processed_image(self, img):
        """
        処理済みの画像を保存するメソッド。
        :param img: 処理済みの画像
        :return: 保存先のファイルパス
        """
        # 画像を保存するディレクトリを定義
        output_dir = Path(__file__).resolve().parent / "processed_images"
        output_dir.mkdir(exist_ok=True)

        # UUIDを使用して一意のファイル名を生成
        import uuid
        file_name = f"processed_{uuid.uuid4().hex}.jpg"
        output_path = output_dir / file_name

        # 画像を保存
        cv2.imwrite(str(output_path), img)

        return str(output_path)


# Falconアプリケーションの初期化
app = falcon.App()

# リソースの登録
happiness_meter = HappinessMeterResource()
app.add_route('/HappinessMeter', happiness_meter)

# サーバーを起動
if __name__ == "__main__":
    server_ip = '127.0.0.1'
    server_port = 8000
    print(f"Server running on http://{server_ip}:{server_port}/HappinessMeter")

    with make_server(server_ip, server_port, app) as httpd:
        httpd.serve_forever()

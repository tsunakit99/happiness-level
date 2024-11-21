import json
from pathlib import Path
from wsgiref.simple_server import make_server

import cv2
import dlib
import falcon
import numpy as np
from image2string import image2string
from string2image import string2image


class HappinessMeterResource:
    def __init__(self):
        # Dlibの顔検出器とランドマーク検出器を初期化
        self.detector = dlib.get_frontal_face_detector()
        # 動的にファイルパスを解決
        model_path = Path(__file__).resolve().parent.parent / "models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.predictor = dlib.shape_predictor(str(model_path))

    def on_post(self, req, resp):
        # POSTされたデータを受け取る
        params = req.media
        img_as_text = params['imagedata']

        # Base64エンコードされた画像をデコード
        img = string2image(img_as_text, False)

        # 幸せ度合いを計算
        happiness_score, processed_image = self.calculate_happiness(img)

        # 処理後の画像をBase64にエンコード
        processed_img_as_text = image2string(processed_image, False)

        # JSON応答を作成
        response_data = {
            "happiness_score": happiness_score,
            "processed_image": processed_img_as_text
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

        # ランドマークから特徴量を計算
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

        # スコアリング
        happiness_score = 50  # 基準スコア

        # 条件分岐を用いたスコアリング
        if normalized_eye_height < 0.03 and normalized_mouth_height > 0.05:
            # 目が細く、口が開いている -> 笑顔
            happiness_score += 20
        elif normalized_eye_height > 0.035 and normalized_mouth_height > 0.05:
            # 目が大きく開き、口が開いている -> 怒りまたは驚き
            happiness_score -= 20
        else:
            # その他の場合、各特徴量に基づいてスコアを調整
            # 口角の角度によるスコア
            if mouth_angle < -15:
                happiness_score += 10
            elif mouth_angle > 15:
                happiness_score -= 10

            # 口の高さと幅の比率によるスコア
            if 0.3 <= mouth_ratio <= 0.6:
                happiness_score += 5
            elif mouth_ratio > 0.6:
                happiness_score -= 5

            # 眉間の距離によるスコア
            if normalized_brow_distance < 0.1:
                happiness_score -= 10

        # デバッグ情報の出力（詳細）
        print(f"Debug Info:")
        print(f"mouth_angle: {mouth_angle:.2f}")
        print(f"normalized_mouth_height: {normalized_mouth_height:.4f}")
        print(f"normalized_mouth_width: {normalized_mouth_width:.4f}")
        print(f"mouth_ratio: {mouth_ratio:.4f}")
        print(f"normalized_eye_height: {normalized_eye_height:.4f}")
        print(f"normalized_brow_distance: {normalized_brow_distance:.4f}")
        print(f"happiness_score: {happiness_score}")

        # スコアの範囲を0から100に制限
        happiness_score = min(max(happiness_score, 0), 100)

        # ランドマークを描画
        for (x, y) in points:
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        return happiness_score, img



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

import base64

import requests

# 画像をBase64にエンコード
with open("../public/example_images/smile2.jpg", "rb") as img_file:
    img_as_text = base64.b64encode(img_file.read()).decode("utf-8")

# APIにリクエストを送信
url = "http://127.0.0.1:8000/HappinessMeter"
data = {"imagedata": img_as_text}
response = requests.post(url, json=data)

# 結果を表示
response_data = response.json()
print(f"Happiness Score: {response_data['happiness_score']}")
import base64

import cv2
import numpy as np


#文字符号化画像データを通常画像データに変換し、returnする
def string2image(arg, flg):
    print('string2image: #文字符号化画像データを通常画像データに変換し、returnする')

    img_as_text=None
    if (flg): # Trueの時はargはファイル名
        # 画像のテキスト形式を読み込み
        with open(arg, mode='r') as rf:
            img_as_text = rf.read()
            print('文字符号化画像データを読み込みました<{}>'.format(rfname))
    else: # Falseの時はargは符号化画像データそのもの
        img_as_text = arg
    
    # 文字列にbyte型に変換
    img_base64=img_as_text.encode('utf-8')

    #base64復号化
    img_binary = base64.b64decode(img_base64)

    #配列に変換
    img_array = np.frombuffer(img_binary, dtype=np.uint8)

    #jpg復号化する
    img_from_text = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img_from_text

if __name__ == "__main__":
        
    #文字符号化画像データを通常画像データ（cv2.imreadで読み込んだ状態）に変換
    rfname='image_string.txt'
    img_from_text = string2image(rfname, True)
    
    #画像データを出力
    wfname='string_image.jpg'
    cv2.imwrite(wfname, img_from_text)
    print('ファイル出力完了<{}>'.format(wfname))


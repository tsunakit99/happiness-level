import base64
import sys

import cv2


#画像データを文字符号化し、returnする
def image2string(arg, flg):
    print('image2string : 画像ファイル名を与えると、画像データを文字符号化し、returnする')

    img = None
    if (flg): # Trueの時はargはファイル名
        img = cv2.imread(arg, cv2.IMREAD_COLOR)
        # img は cv2.imread などで得られる画像を読み込んだnumpy.ndarray

        # 読み込み状況の確認
        if img is None:
            print('<{}>　正しく読み込めませんでした'.format(arg), file=sys.stderr)
            sys.exit(1)
        else:
            print('画像データを読み込みました<{}>'.format(arg))
    else: # Falseの時はargは画像データそのもの
        img = arg
        
    #imgをjpg符号化しバイトデータをbuffer変数に入れる
    _, buffer = cv2.imencode('.jpg', img)

    # ビット系列を素直にファイル出力
    wfname='temporary.jpg'
    with open(wfname, mode='wb') as wf:
        wf.write(buffer)
    print('バイナリデータを書き込みました<{}>'.format(wfname))
    
    #bufferをbase64符号化する
    img_base64=base64.b64encode(buffer)

    # base64のbyte文字列(img_base64)をdecodeで文字列に変換
    img_as_text = img_base64.decode('utf-8')

    return img_as_text

if __name__ == "__main__":
    rfname='fullcolor_sample.jpg'
    img_as_text = image2string(rfname, True)
    
    # テキストファイルとして出力
    wfname='image_string.txt'
    with open(wfname, mode='w') as wf:
        wf.write(img_as_text)
    print('符号化画像データを書き込みました<{}>'.format(wfname))

    

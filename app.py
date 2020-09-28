from flask import Flask, send_file
import cairosvg
import cv2
import os


app = Flask(__name__)


def mosaicing(src, ratio=0.01):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)


def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaicing(dst[y:y + height, x:x + width], ratio)
    return dst


@app.route('/')
def mosaic():
    url = 'https://joeschmoe.io/api/v1/random'

    cairosvg.svg2png(url=url, write_to='org.png')

    try:
        cairosvg.svg2png(
            url=url, write_to='org.png')
    except Exception as e:
        print(e)
        cairosvg.svg2png(
            url=url, write_to='org.png')

    img = cv2.imread('org.png')

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_position = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in face_position:
        mosaic_face = mosaic_area(img, x, y, w, h)

    img = cv2.putText(img, 'Org', (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
    mosaic_faceText = cv2.putText(mosaic_face, 'mosaic', (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)

    concat_img = cv2.hconcat([img, mosaic_faceText])

    cv2.imwrite('generated.png', concat_img)

    return send_file('generated.png', mimetype='image/jpg')


if __name__ == "__main__":
     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

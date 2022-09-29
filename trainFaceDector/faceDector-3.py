# 人脸识别
# coding=utf-8
import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont
# 發布者（publisher）指令稿 publisher1.py
import paho.mqtt.client as mqtt

# 建立 MQTT Client 物件
#client = mqtt.Client()
# 連線至 MQTT 伺服器（伺服器位址,連接埠）
#client.connect("192.168.76.155", 1883)

# 解决cv2.putText绘制中文乱码
def cv2ImgAddText(img2, text, left, top, textColor=(0, 0, 255), textSize=20):
    if isinstance(img2, numpy.ndarray):  # 判断是否OpenCV图片类型
        img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img2)
    # 字体的格式
    fontStyle = ImageFont.truetype(r"C:\WINDOWS\FONTS\MSYH.TTC", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img2), cv2.COLOR_RGB2BGR)

def face_detection(cv2ImgAddText):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('AIot/train/train.yml')
    cascadePath = "AIot/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    num = 0
    names = ['不認識的臉孔', '邱定凱']
    cam = cv2.VideoCapture(0)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH))
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            num, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            name = names[num]
            if confidence < 100:
                if name=="邱定凱":
                    #client.publish("door/opening","1")
                    print(name)
                # confidence = "{0}%".format(round(100 - confidence))
                # confidence = format(round(100 - confidence))
            else:
                name == "不認識的臉孔"
                #client.publish("door/opening","0")
                print(name)
                # confidence = "{0}%".format(round(100 - confidence))
                # confidence = format(round(100 - confidence))

            # 解决cv2.putText绘制中文乱码
            img = cv2ImgAddText(img, name, x + 5, y - 30)
            # cv2.putText(img, name, (x + 5, y - 5), font, 1, (0, 0, 255), 1) 无法显示中文
            # cv2.putText(img, str(confidence.encode('utf-8')), (x+5, y+h-5), font, 1, (0, 0, 0), 1)

        cv2.imshow('camera', img)
        k = cv2.waitKey(5)
        if k == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    face_detection(cv2ImgAddText)
import cv2
import numpy as np
import json
from darkflow.darkflow.net.build import TFNet
from flask import Flask, request

app = Flask(__name__)

options = {"model": "./darkflow/cfg/my-tiny-yolo.cfg",
           "pbLoad": "./darkflow/darkflow/built_graph/my-tiny-yolo.pb",
           "metaLoad": './darkflow/darkflow/built_graph/my-tiny-yolo.meta', "threshold": 0.4
           }
tfnet = TFNet(options)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/yolo', methods=['GET'])
def yolo():
    # 업로드 폴더에 있는 해당 이미지 읽기
    img = cv2.imread('./upload/test.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = []

    # tfnet 라이브러리를 이용하여 해당 이미지의 결과 json으로 뽑기
    results = tfnet.return_predict(img)
    print(results)
    for result in results:
        arr.append(json.dumps(str(result))) #json.dumps 해야 형변환 해서 백엔드에 보냄

    return {
        'yolo_result': arr
    }


if __name__ == '__main__':
    app.run()

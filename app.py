import cv2
import numpy as np
import sys
import time
import requests
import json
import urllib.request
from collections import OrderedDict
from darkflow.darkflow.net.build import TFNet
from flask import Flask, request, jsonify

app = Flask(__name__)

options = {"model": "./darkflow/cfg/my-tiny-yolo.cfg",
           "pbLoad": "./darkflow/darkflow/built_graph/my-tiny-yolo.pb",
           "metaLoad": './darkflow/darkflow/built_graph/my-tiny-yolo.meta', "threshold": 0.4
           }
tfnet = TFNet(options)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/yolo', methods=['POST'])
def yolo():
    filenames = request.args["file_name"]
    filenames = filenames.split(',')
    del filenames[0] #workbook_sn 삭제
    to_node = []
    for filename in filenames:
        print(filename)
        #http 링크로 하는거
        req = urllib.request.urlopen(filename)
        img = np.asarray(bytearray(req.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # 업로드 폴더에 있는 해당 이미지 읽기
        #img = cv2.imread(img_l, cv2.IMREAD_COLOR)
        #img = cv2.imread('./upload/' + filename + '.jpg', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = []

        # tfnet 라이브러리를 이용하여 해당 이미지의 결과 json으로 뽑기
        results = tfnet.return_predict(img)
        print(results)
        for result in results:
            arr.append(json.dumps(str(result)))   #json.dumps 해야 형변환 해서 백엔드에 보냄
        to_node.append(arr)

    return jsonify({
        'yolo_result': to_node
    })


@app.route('/test', methods=['POST'])
def test():
    lists = request.args['file_name']
    lists = lists.split(',')
    data = []
    for list in lists:
        data.append(list)

    return jsonify({
        'result': data
    })


if __name__ == '__main__':
    app.run()

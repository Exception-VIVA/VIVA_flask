import cv2
import numpy as np
import sys
import time
import requests
import json
import urllib.request
from collections import OrderedDict
from flask import Flask, request, jsonify

app = Flask(__name__)

LIMIT_PX = 1024
LIMIT_BYTE = 1024 * 1024  # 1MB
LIMIT_BOX = 40


def kakao_ocr_resize(image_path: str):
    """
    ocr detect/recognize api helper
    ocr api의 제약사항이 넘어서는 이미지는 요청 이전에 전처리가 필요.

    pixel 제약사항 초과: resize
    용량 제약사항 초과  : 다른 포맷으로 압축, 이미지 분할 등의 처리 필요. (예제에서 제공하지 않음)

    :param image_path: 이미지파일 경로
    :return:
    """
    image = image_path

    height, width, _ = image.shape

    if LIMIT_PX < height or LIMIT_PX < width:
        ratio = float(LIMIT_PX) / max(height, width)
        image = cv2.resize(image, None, fx=ratio, fy=ratio)
        height, width, _ = height, width, _ = image.shape

        # api 사용전에 이미지가 resize된 경우, recognize시 resize된 결과를 사용해야함.
        image_path = "{}_resized.jpg".format(image_path)
        cv2.imwrite(image_path, image)

        return image_path
    return None


def kakao_ocr(image_path: str, appkey: str):
    """
    OCR api request example
    :param image_path: 이미지파일 경로
    :param appkey: 카카오 앱 REST API 키
    """
    API_URL = 'https://dapi.kakao.com/v2/vision/text/ocr'

    headers = {'Authorization': 'KakaoAK {}'.format(appkey)}

    image = image_path
    jpeg_image = cv2.imencode(".jpg", image)[1]
    data = jpeg_image.tobytes()

    return requests.post(API_URL, headers=headers, files={"image": data})


def read_ocr(image):
    # if len(sys.argv) != 3:
    #     print("Please run with args: $ python example.py /path/to/image appkey")
    appkey = 'c858cbb7294b2c96b1287054dd31337f'
    white = [255, 255, 255]
    img = image
    # 여기까지 됨
    # image_path = image
    constant = cv2.copyMakeBorder(img, 200, 100, 100, 100, cv2.BORDER_CONSTANT, value=white)
    image_path = constant
    time.sleep(2)
    resize_impath = kakao_ocr_resize(image_path)
    if resize_impath is not None:
        image_path = resize_impath
        print("using resized image")

    output = kakao_ocr(image_path, appkey).json()

    # 하나의 이미지에서 여러개를 읽어냄  -> list에 저장해서 통째로 넘겨줌
    word_list = []
    # 결과내에서 하나씩 골라서
    for out in (output['result']):
        # recognition_word dic부분을 추출해 append함
        outword = out['recognition_words'][0]
        word_list.append(outword)
    return word_list


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/yolo', methods=['POST'])
def yolo():
    filenames = request.args["file_name"]
    filenames = filenames.split(',')
    del filenames[0]  # workbook_sn 삭제
    to_node = []

    for filename in filenames:
        print(filename)
        # http 링크로 하는거
        # 이미지 불러오기
        req = urllib.request.urlopen(filename)
        img = np.asarray(bytearray(req.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # 업로드 폴더에 있는 해당 이미지 읽기
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Yolo 로드
        net = cv2.dnn.readNet("version3/viva-yolov3-tiny-detection-v2_48000.weights",
                              "version3/viva-yolov3-tiny-detection-v2.cfg")
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        output_json = []

        # 정보를 화면에 표시
        # class_ids = []
        # confidences = []
        # boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.15:
                    output = dict()
                    label = str(classes[class_id])
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    output["label"] = label
                    output["confidence"] = float(confidence)
                    output["x"] = int(x)
                    output["y"] = int(y)
                    output["w"] = int(w)
                    output["h"] = int(h)
                    output["recognition_word"] = 'null'

                    # 이거다 원형 -> darkflow에서 제공하던 버전이랑 똑같음
                    json.dumps(output)
                    output_json.append(output)

        # 2. 번호순서대로 정렬을 위해서 반으로 나누어 y축으로 정렬한다.

        # x축을 기준으로 정렬함
        result_fix2 = sorted(output_json, key=lambda result: (result['x'] + result['w']))

        # 왼쪽페이지 json을 저장할 list
        left = []

        # 오른쪽페이지 json을 저장할 list
        right = []

        # 왼쪽/오른쪽을 나눌 x좌표
        half = img.shape[1]

        # x축을 기준으로 정렬한 것을 토대로 절반을 기준으로 오른페이지 왼페이지로 나눔
        for result in (result_fix2):

            if (result['x'] + result['w'] < half / 2):
                left.append(result)
            else:
                right.append(result)

        # 출력해서 확인
        left = sorted(left, key=lambda result: (result['y'] + result['h']))
        right = sorted(right, key=lambda result: (result['y'] + result['h']))

        # 2-3. left, right로 나누어진 것을 하나의 json으로 합침
        left.extend(right)

        result_fix3 = left
        # 3. 중복 detection을 제거
        # 지금 오류있음 -> 저거 딜리트되면 reuslt_fix땡겨져서 하나 넘어가짐;

        index = 0

        for result in result_fix3[:]:
            if (index == 0):
                label_past = result['label']
                size_past = ((result['x'] + result['w'] + result['h'] - result['x']) ** 2 + (
                        result['y'] - result['y'] + result['w'] + result['h']) ** 2) ** 0.5

            else:
                label_now = result['label']
                size_now = ((result['x'] + result['w'] + result['h'] - result['x']) ** 2 + (
                        result['y'] - result['y'] + result['w'] + result['h']) ** 2) ** 0.5

                # OCR을 적용할 spn/short_ans/page_num에만 적용될 예정
                if (result['label'] != 'check_box' and result['label'] != 'uncheck_box'):
                    # 이미 정렬되었으므로 연속으로 나오는 same label은 중복 detection으로 간주
                    if (label_now == label_past):
                        # 크기가 더 작은 쪽을 제거함 -> 제거한 것을 출력해서 확인함
                        if (size_now < size_past):
                            del (result_fix3[index])
                        else:
                            del (result_fix3[index - 1])
                        index = index - 1

                label_past = label_now
                size_past = size_now
            index = index + 1

        # ocr적용해서 읽어들인 내용 수정
        flag = True
        for result in result_fix3[:]:
            if (result['label'] == 'short_ans'):
                x = result['x']
                y = result['y']
                w = result['w']
                h = result['h']
                cropped_img = img[y - 15:y + h + 15, x - 15:x + w + 15]
                recognition_words = read_ocr(cropped_img)
                result["recognition_word"] = recognition_words
            elif (result['label'] == 'spn' and flag == True):
                x = result['x']
                y = result['y']
                w = result['w']
                h = result['h']
                flag = False
                cropped_img = img[y - 15:y + h + 15, x - 15:x + w + 15]
                recognition_words = read_ocr(cropped_img)
                result["recognition_word"] = recognition_words

        to_node.append(result_fix3)

    return jsonify({
        "yolo_result": to_node
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

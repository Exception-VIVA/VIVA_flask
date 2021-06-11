# YOLOv3



최종 버전 : version5

사용자가 촬영한 시험지 이미지에서 채점에 필요한 요소를 찾아내는 CNN 모델



## Image Label

label 도구 

[tzutalin/labelImg](https://github.com/tzutalin/labelImg.git)

txt파일로 변환

[Isabek/XmlToTxt](https://github.com/Isabek/XmlToTxt)

Class

- sn_sw : "수능 완성"의 고유번호(serial number)
- pn_sw : "수능완성의 문제번호(problem number)
- spn : "교육청/평가원" 고유번호인 동시에 문제번호
- check_box : 객관식(multiple choices) 답으로 체크한 것
- uncheck_box : 객관식(multiple choices) 답으로 체크되지 않은 것
- short_ans :  주관식 네모
- sw_page_num : 수완 페이지 번호
- page_num : 평가/교육청 페이지 번호

![라벨링예시](https://user-images.githubusercontent.com/74401770/121140964-69ff4d00-c875-11eb-90b8-f3774a6c5444.png)

라벨링 예시 이미지



## YOLO model

YOLO v3 model 

[Meet Google Drive - One place for all your files](https://drive.google.com/drive/folders/1eOwbsDb3TOxuwKgGquPRibpeFupyEHU8?usp=sharing)

용량 제한으로 인한 드라이브 링크 업로드 

## Training Data set

https://drive.google.com/drive/folders/1_jFmNMwpSq_7FyYI7VtYmvvBxcptDpyI?usp=sharing

모델학습에 사용한 데이터 모음



## CODE

실제 채점시 형식 설정을 위한 전처리 코드



### 라이브러리

```python
cv2
numpy
sys
time
request
json
```

 
 
### Preprocessing

1. YOLO detecting
2. 문제번호 순서대로 요소 정렬
3. 중복제거
4. ocr을 이용한 문자인식 → udpate . ocr을 첫번째 spn에만 적용하는 것으로 수정(시간 오래걸림)

```python
# 이미지 불러오기-> 시각화용도로 5개 
img = cv2.imread("scan/2019_sn_ga/2019_sn_ga_9.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Yolo 로드
net = cv2.dnn.readNet("version5/viva-yolov3-tiny-detection-v2_final.weights", "version5/viva-yolov3-tiny-detection-v2.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

#OCR적용 

LIMIT_PX = 1024
LIMIT_BYTE = 1024*1024  # 1MB
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
    if len(sys.argv) != 3:
        print("Please run with args: $ python example.py /path/to/image appkey")
    appkey = 'c858cbb7294b2c96b1287054dd31337f'
    white = [255,255,255]
    img=image
    image_path = image
    constant= cv2.copyMakeBorder(img,200,100,100,100,cv2.BORDER_CONSTANT,value=white)
    #image_path='/content/drive/MyDrive/GRADING_Study/kh/hand/ex/sw.jpeg'
    #cv2.imwrite(image_path, constant)
    #cv2_imshow(constant)
    image_path=constant
    time.sleep(2)
    resize_impath = kakao_ocr_resize(image_path)
    if resize_impath is not None:
        image_path = resize_impath
        print("using resized image")

    #cv2.imshow("OCR_Input",image_path)
    #k = cv2.waitKey(0)
    #cv2.destroyWindow("OCR_Input")
        
    output = kakao_ocr(image_path, appkey).json()
    #print("[OCR] output:\n{}\n".format(json.dumps(output, sort_keys=True,ensure_ascii=False, indent=2)))
    
    
    #하나의 이미지에서 여러개를 읽어냄  -> list에 저장해서 통째로 넘겨줌 
    word_list=[] 
    # 결과내에서 하나씩 골라서 
    for out in (output['result']):
        #recognition_word dic부분을 추출해 append함 
        outword = out['recognition_words'][0]
        word_list.append(outword)
    #print(word_list)
    #words = output['result'][0]['recognition_words'][0]
    #print("======OUTPUT======")
    #print(output)
    #print("========END=====")
    #리스트 return
    return word_list

output_json = []

# 정보를 화면에 표시

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.15:
            output = dict()
            label = str(classes[class_id])
            # Object detected
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
         
              
    
            #이거다 원형 -> darkflow에서 제공하던 버전이랑 똑같음 
            json.dumps(output)
            output_json.append(output)
            
#2. 번호순서대로 정렬을 위해서 반으로 나누어 y축으로 정렬한다. 

#x축을 기준으로 정렬함 
result_fix2 = sorted(output_json, key=lambda result: (result['x']+result['w']))

#왼쪽페이지 json을 저장할 list
left=[]

#오른쪽페이지 json을 저장할 list
right=[]

#왼쪽/오른쪽을 나눌 x좌표 
half = img.shape[1]
#print(half)

#x축을 기준으로 정렬한 것을 토대로 절반을 기준으로 오른페이지 왼페이지로 나눔 
for result in (result_fix2):
    
    if(result['x']+result['w']<half/2):
        left.append(result)
    else:
        right.append(result)
        

#출력해서 확인 
left = sorted(left, key=lambda result: (result['y']+result['h']))
right = sorted(right, key=lambda result: (result['y']+result['h']))

#2-3. left, right로 나누어진 것을 하나의 json으로 합침 
left.extend(right)

result_fix3 = left

#3. 중복 detection을 제거 

index = 0

for result in result_fix3[:]:
    if(index == 0):
        label_past = result['label']
        size_past = ((result['x']+result['w']+result['h']-result['x'])**2 + (result['y']-result['y']+result['w']+result['h'])**2)**0.5
        #print('<-label->')
        #print(label_past)
    else:
        label_now = result['label']
        size_now = ((result['x']+result['w']+result['h']-result['x'])**2 + (result['y']-result['y']+result['w']+result['h'])**2)**0.5
        #print("<-label->")
        #print(label_now)
        #OCR을 적용할 spn/short_ans/page_num에만 적용될 예정 
        if (result['label']!='check_box' and result['label']!='uncheck_box'):
        #if(label_now=='spn' or label_now=='short_ans' or label_now=='page_num'):
            #이미 정렬되었으므로 연속으로 나오는 same label은 중복 detection으로 간주 
            if(label_now == label_past):
                print("same")
                print(label_now)
                print(label_past)
                #크기가 더 작은 쪽을 제거함 -> 제거한 것을 출력해서 확인함 
                if(size_now<size_past):
                    del(result_fix3[index])
                else :
                    del(result_fix3[index-1])
                index=index-1    
        label_past = label_now
        size_past = size_now
    index = index + 1

#ocr적용해서 읽어들인 내용 수정 -> update 주관식과 페이지 첫 문제번호(spn)만 인식하도록 변경
flag = True
for result in result_fix3[:]:
    if(result['label']=='short_ans'):
        cropped_img = img[y-15:y+h+15, x-15:x+w+15]
        recognition_words = read_ocr(cropped_img)
        result["recognition_word"] = recognition_words
    elif(result['label']=='spn' and flag == True):
        flag = False
        cropped_img = img[y-15:y+h+15, x-15:x+w+15]
        recognition_words = read_ocr(cropped_img)
        result["recognition_word"] = recognition_words

        
print("=====final=======")
print(result_fix3)
```

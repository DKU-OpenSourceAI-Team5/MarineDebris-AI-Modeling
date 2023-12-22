# 🌊 MarineDebris-AI-Modeling
### 해양 쓰레기 이미지 분류 AI 학습 모델 구현
- 해양 쓰레기 이미지 데이터를 종류 별로 분류할 수 있는 딥러닝 모델을 구현하여<br>추후 해안/해양 쓰레기 자동 탐지 프로그램, 해양 쓰레기 분포도 생성 프로그램 등에 활용 가능하도록 한다.<br>
- ver.1(Colab 실행 코드)와 ver.2(Pycharm 실행 코드) 구현

<br>

## 🤖 Model 설명
### CNN 모델(Convolutional Neural Network, 합성곱 신경망)
> 이미지 인식 및 패턴 인식과 같은 작업에 특화된 신경망 구조<br>이미지의 특징을 학습하여 이미지 분류, 객체 검출, 세그멘테이션 등의 작업을 수행하는 데에 활용되는 딥러닝 모델


- 주요 구성 요소
  1. 합성곱 층 (Convolutional Layer):<br>
  합성곱 연산: 이미지에 커널(필터)을 이동시키며 지역적인 특징을 감지한다. 각 합성곱 연산은 특정한 패턴이나 엣지와 같은 시각적 정보를 추출한다.<br>
  스트라이드(Striding): 커널을 이동시키는 간격을 의미하며, 스트라이드가 클수록 출력 크기는 작아진다.
  2. 활성화 함수 (Activation Function):<br>
  ReLU(Rectified Linear Unit): 주로 사용되는 활성화 함수로, 입력이 양수이면 그 값을 그대로 사용하고, 음수일 경우에는 0으로 만든다. 비선형성을 도입하여 모델이 더 복잡한 패턴을 학습할 수 있도록 하는 함수이다.
  3. 풀링 층 (Pooling Layer):<br>
  풀링 연산: 공간 차원을 줄이기 위해 사용되며, 주로 최대 풀링(Max Pooling)이나 평균 풀링(Average Pooling)이 적용된다. 이를 통해 계산 비용을 감소시키고 중요한 특징을 강조할 수 있다.
  4. 완전 연결 층 (Fully Connected Layer):<br>
  Flatten 층: CNN의 출력을 1차원으로 펼친다.<br>
  완전 연결 층: 펼친 결과를 입력으로 받아 최종적인 분류나 회귀를 수행한다.
  5. 손실 함수 (Loss Function) 및 최적화:<br>
  손실 함수: 모델의 예측과 실제 값 사이의 차이를 측정하는 함수로, 분류 문제에서는 주로 crossentropy 손실이 사용된다. 따라서 해당 모델에서도 crossentropy가 사용되었다.<br>
  최적화 알고리즘: 경사 하강법(GD)의 변형으로, 역전파 알고리즘을 사용하여 가중치를 업데이트하고 손실을 최소화한다. 해당 모델에서는 adam을 사용하였다. 


<br>

## 📙 ver.1: Colab 실행 코드
### 기본 정보
- 실행 OS: Window 11
- python version: 3.10.12
- required libraries: json, pathlib, numpy, tensorflow, keras, sklearn, PIL


### 실행 방법
1. Import Libraries
```python
# 코랩 드라이브 연결하기
from google.colab import drive
drive.mount('/content/drive')

# 필요한 라이브러리 및 모듈 불러오기
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
from keras import models, layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
```
2. Data Load
```python
# JSON 파일들이 있는 디렉토리 경로
json_directory_path = '/content/drive/MyDrive/ColabNotebooks/2023/opensource/trash_label/'

# 이미지 파일들이 있는 디렉토리 경로
image_directory_path = '/content/drive/MyDrive/ColabNotebooks/2023/opensource/trash_image/'

# 이미지와 라벨을 저장할 리스트
images = []
labels = []

# 지정된 디렉토리에서 JSON 파일을 읽어오기 위한 파일 리스트 생성
data_size = 1000
json_files = list(Path(json_directory_path).rglob('*.json'))[:data_size]

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

        # 이미지 경로
        image_path = Path(image_directory_path) / data['imagePath']

        # 이미지 불러오기
        image = np.array(Image.open(image_path))

        # 라벨 정보 추출
        shapes = data.get('shapes', [])
        for shape in shapes:
            label = shape['label']
            points = shape['points']

            images.append(image)
            labels.append(label)
```
3. Pre Processing
```python
# 이미지와 라벨을 넘파이 배열로 변환
images = np.array(images)
labels = np.array(labels)

# 라벨을 정수로 변환 (라벨 인코딩)
label_binarizer = LabelBinarizer()
labels_encoded = label_binarizer.fit_transform(labels)
```
4. Data Split
```python
# 데이터 분할
train_images, test_images, train_labels, test_labels = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# 이미지 크기 조정
target_image_size = (224, 224)
train_images_resized = [tf.image.resize(image, target_image_size) for image in train_images]
test_images_resized = [tf.image.resize(image, target_image_size) for image in test_images]

# 이미지를 TensorFlow 텐서로 변환
train_images_tensor = tf.convert_to_tensor(train_images_resized, dtype=tf.float32)
test_images_tensor = tf.convert_to_tensor(test_images_resized, dtype=tf.float32)

# 라벨을 TensorFlow 텐서로 변환
train_labels_tensor = tf.convert_to_tensor(train_labels, dtype=tf.float32)
test_labels_tensor = tf.convert_to_tensor(test_labels, dtype=tf.float32)
```
5. Define Model - CNN
```python
# CNN 모델 생성
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(target_image_size[0], target_image_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(label_binarizer.classes_), activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
6. Model Training
```python
# 모델 훈련
epochs = 5
batch_size = 16

history = model.fit(
    train_images_np, train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test_images_np, test_labels)
)
```
7. Model Test
```python
# 정확도 확인
test_loss, test_accuracy = model.evaluate(test_images_np, test_labels)
print('\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
print(f'Dataset Size: {data_size}')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
```

<br>

## 📙 ver.2: Pycharm(conda) 실행 코드
### 기본 정보
- 실행 OS: macOS
- python version: 3.11
- required libraries: numpy, tensorflow, sklearn(scikit-learn), PIL(Pillow)


### 실행 방법
1. Import Libraries
```python
# 필요한 라이브러리 및 모듈 불러오기
import json  # JSON 파일 처리
from pathlib import Path  # 파일 경로 관리
import numpy as np  # 배열 처리
import tensorflow as tf  # 딥러닝 라이브러리
from sklearn.model_selection import train_test_split  # 데이터 분할
from sklearn.preprocessing import LabelBinarizer  # 라벨 인코딩
from PIL import Image  # 이미지 처리
from keras.models import Sequential  # Sequential 모델
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # CNN 레이어들
```
2. Data Load
```python
# JSON 파일과 이미지 파일이 있는 디렉토리 경로 설정
json_directory_path = './data/label/'
image_directory_path = './data/image/'

# 지정된 디렉토리에서 JSON 파일을 읽어오기 위한 파일 리스트 생성
data_size = 1500
json_files = list(Path(json_directory_path).rglob('*.json'))[:data_size]

# 데이터를 저장할 리스트 초기화
data_pairs = []

# JSON 파일들을 순회하며 이미지와 라벨 정보 추출
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

        # 이미지 경로 설정
        image_path = Path(image_directory_path) / data['imagePath']

        # 이미지 불러오기 및 전처리
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)

        # 라벨 정보 추출하여 데이터 페어에 추가
        shapes = data.get('shapes', [])
        for shape in shapes:
            label = shape['label']
            data_pairs.append((image_array, label))
```
3. Pre Processing
```python
# 이미지와 라벨을 각각 NumPy 배열로 변환
images = np.array([pair[0] for pair in data_pairs])
labels = np.array([pair[1] for pair in data_pairs])

# 라벨을 정수로 변환 (라벨 인코딩)
label_binarizer = LabelBinarizer()
labels_encoded = label_binarizer.fit_transform(labels)

# 이미지 크기 조정
target_image_size = (224, 224)
images_resized = [tf.image.resize(image, target_image_size) for image in images]

# 이미지를 TensorFlow 텐서로 변환
images_tensor = tf.convert_to_tensor(images_resized, dtype=tf.float32)

# TensorFlow 텐서를 NumPy 배열로 변환
images_np = images_tensor.numpy()
```
4. Data Split
```python
# 데이터 분할 (훈련 데이터와 테스트 데이터)
train_images_np, test_images_np, train_labels, test_labels = train_test_split(images_np, labels_encoded, test_size=0.2, random_state=42)
```
5. Define Model - CNN
```python
# CNN 모델 생성
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(target_image_size[0], target_image_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(label_binarizer.classes_), activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
6. Model Training
```python
# 모델 훈련
epochs = 5
batch_size = 16

history = model.fit(
    train_images_np, train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test_images_np, test_labels)
)
```
7. Model Test
```python
# 정확도 확인
test_loss, test_accuracy = model.evaluate(test_images_np, test_labels)
print('\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
print(f'Dataset Size: {data_size}')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
```

<br>

## 📊 실행 결과
### ver.1 결과
&nbsp;&nbsp;**1. Dataset 크기 = 500**<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Loss: 1.6855<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Accuracy: 0.4420

<img width="658" alt="KakaoTalk_Photo_2023-12-22-20-17-06 001png" src="https://github.com/DKU-OpenSourceAI-Team5/MarineDebris-AI-Modeling/assets/86196342/6f8828e5-6aea-4462-bf38-66064ba673fc"><br>



&nbsp;&nbsp;**2. Dataset 크기 = 1000**<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Loss: 1.1954<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Accuracy: 0.5631

<img width="680" alt="KakaoTalk_Photo_2023-12-22-20-17-06 002png" src="https://github.com/DKU-OpenSourceAI-Team5/MarineDebris-AI-Modeling/assets/86196342/5eff2e36-fb21-4350-b1ef-32cb6c180cea"><br>



&nbsp;&nbsp;**3. Dataset 크기 = 1500**<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Loss: 1.1089<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Accuracy: 0.5704

<img width="660" alt="KakaoTalk_Photo_2023-12-22-20-17-06 003png" src="https://github.com/DKU-OpenSourceAI-Team5/MarineDebris-AI-Modeling/assets/86196342/4e20f0a6-9721-42ff-a223-cdee266b1b6e"><br>



<br>

### ver.2 결과
&nbsp;&nbsp;**1. Dataset 크기 = 500**<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Loss: 1.8764<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Accuracy: 0.4229

<img width="1202" alt="pycharm_500" src="https://github.com/DKU-OpenSourceAI-Team5/MarineDebris-AI-Modeling/assets/86196342/c887ad8c-f56a-48ba-951d-b4aee2fb5b1b"><br>


&nbsp;&nbsp;**2. Dataset 크기 = 1000**<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Loss: 1.7778<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Accuracy: 0.4191

<img width="1145" alt="pycharm_1000" src="https://github.com/DKU-OpenSourceAI-Team5/MarineDebris-AI-Modeling/assets/86196342/710a9241-727c-44b0-9d2f-210600698420"><br>


&nbsp;&nbsp;**3. Dataset 크기 = 1500**<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Loss: 1.8276<br>
&nbsp;&nbsp;&nbsp;&nbsp;▶️ Total Accuracy: 0.4520

<img width="1146" alt="pycharm_1500" src="https://github.com/DKU-OpenSourceAI-Team5/MarineDebris-AI-Modeling/assets/86196342/3774c58c-b076-45f8-8d50-8df2c1c7a945">


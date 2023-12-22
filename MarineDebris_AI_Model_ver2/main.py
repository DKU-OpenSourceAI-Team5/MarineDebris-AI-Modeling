# -*- coding: utf-8 -*-

# 해양쓰레기 분류 학습 AI 모델: Pycharm 실행 ver.

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

# 데이터 분할 (훈련 데이터와 테스트 데이터)
train_images_np, test_images_np, train_labels, test_labels = train_test_split(images_np, labels_encoded, test_size=0.2, random_state=42)

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

# 모델 훈련
epochs = 5
batch_size = 16

history = model.fit(
    train_images_np, train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test_images_np, test_labels)
)

# 정확도 확인
test_loss, test_accuracy = model.evaluate(test_images_np, test_labels)
print('\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
print(f'Dataset Size: {data_size}')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

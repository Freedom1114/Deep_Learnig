import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 모델 로드
model = load_model('./Cat vs Dog/cat_dog_classifier.h5')

# 테스트 이미지 폴더 경로
test_dir = 'C:/Users/USER/Desktop/data/test'

# 테스트 폴더의 모든 이미지 파일 가져오기
test_images = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# 각 이미지 테스트
for img_path in test_images:
    try:
        # 이미지 전처리
        test_image = image.load_img(img_path, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0) / 255.0

        # 예측
        prediction = model.predict(test_image)
        if prediction[0] > 0.5:
            print(f"The image at {img_path} is a Dog.")
        else:
            print(f"The image at {img_path} is a Cat.")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, ViTModel, ViTFeatureExtractor
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import requests
from io import BytesIO

# pm_code에 대한 설명 사전
pm_code_descriptions = {
    30: "킥보드 탑승자가 보행자도로 통행 위반",
    31: "킥보드 탑승자가 안전모 미착용 위반",
    32: "킥보드 탑승자가 무단횡단 위반",
    33: "킥보드 탑승자가 신호 위반",
    35: "킥보드 탑승자가 횡단보도 주행 위반",
    36: "킥보드 탑승자가 동승자 탑승 위반"
}

# 1. 학습된 토크나이저 및 모델 불러오기
kogpt2_model_path = './model'  # 저장된 모델 경로
kogpt2_model = AutoModelForCausalLM.from_pretrained(kogpt2_model_path)  # 학습된 KoGPT-2 모델
kogpt2_tokenizer = AutoTokenizer.from_pretrained(kogpt2_model_path)  # 학습된 KoGPT-2 토크나이저

# 2. ViT 모델 불러오기
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# 3. 분류 모델 정의
class PMCodeClassifier(nn.Module):
    def __init__(self, vit_model, num_labels):
        super(PMCodeClassifier, self).__init__()
        self.vit_model = vit_model
        self.fc = nn.Linear(vit_model.config.hidden_size, num_labels)  # pm_code 예측을 위한 fully connected layer

    def forward(self, image_input):
        vit_outputs = self.vit_model(pixel_values=image_input)
        vit_embeddings = vit_outputs.last_hidden_state[:, 0, :]
        logits = self.fc(vit_embeddings)
        return logits

# 4. 이미지 전처리 함수
def preprocess_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs['pixel_values']

# 5. 새로운 이미지에 대한 pm_code 예측 및 설명 생성 함수
def generate_description_for_image(image_path, classifier):
    # 이미지 전처리
    image_input = preprocess_image(image_path)

    # pm_code 예측
    with torch.no_grad():
        logits = classifier(image_input)
        pm_code_pred = torch.argmax(logits, dim=-1).item()  # 예측된 pm_code

    # 예측된 pm_code에 해당하는 설명 선택
    pm_code = list(pm_code_descriptions.keys())[pm_code_pred]
    description = pm_code_descriptions[pm_code]

    return description

# 6. 학습된 분류 모델 정의
num_labels = len(pm_code_descriptions)
classifier = PMCodeClassifier(vit_model, num_labels)

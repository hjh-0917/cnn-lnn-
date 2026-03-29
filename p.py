import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_path = 'C:/Users/jeonghyeon/algorithem/Vocabulary.json'
model_path = 'C:/Users/jeonghyeon/algorithem/p1.pth'
test_image_path = "C:/Users/jeonghyeon/Desktop/다운로드 (2).jpg"

with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

idx_to_token = list(vocab.keys())
token_id_list = list(vocab.values())
word_count = len(idx_to_token)

class algorithem1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.image_to_features = nn.Linear(64, 768)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.word_embedding = nn.Embedding(word_count, 768)
        self.llm_model = nn.Sequential(
            nn.Linear(768, word_count)
        )

    def forward(self, img, previous_embedding_id):
        img_feat = self.cnn_model(img)
        img_feat = img_feat.flatten(2).permute(2, 0, 1)
        img_feat = self.image_to_features(img_feat)
        word_feat = self.word_embedding(previous_embedding_id).unsqueeze(0)
        attn_output, _ = self.attention(query=word_feat, key=img_feat, value=img_feat)
        combined = attn_output.squeeze(0) + word_feat.squeeze(0)
        return self.llm_model(combined)

model = algorithem1().to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("가중치 로드 성공")
else:
    print("가중치 파일 없음")

model.eval()

transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor()
])

def predict_caption(image_path, max_length=10000):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    current_token_index = torch.tensor([0], dtype=torch.long).to(device)

    print("예측 결과: ", end="", flush=True)

    with torch.no_grad():
        for _ in range(max_length):
            output = model(img, current_token_index)
            predict_id = output.argmax().item()
            predict_word = idx_to_token[predict_id]

            if predict_word == "SEP":
                print()  # 줄바꿈
                break

            print(predict_word, end=" ", flush=True)
            current_token_index = torch.tensor([predict_id], dtype=torch.long).to(device)

if __name__ == "__main__":
    print(f"입력 이미지: {os.path.basename(test_image_path)}")
    predict_caption(test_image_path)
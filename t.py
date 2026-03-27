import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import time
import os
import torchvision.transforms as transforms
import json
from data import d
import random

#JSON 불러오기
with open('C:/Users/jeonghyeon/algorithem/Vocabulary.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
token_id_list = list(vocab.values())

with open('C:/Users/jeonghyeon/algorithem/revocabulary.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(data)

idx_to_token = list(vocab.keys())  
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
            nn.Linear(768 ,word_count )
        )


    def forward(self,img, previous_embedding_id):
        img_feat = self.cnn_model(img)
        img_feat = img_feat.flatten(2).permute(2, 0, 1)
        img_feat = self.image_to_features(img_feat) # (14400, 1, 768)

        word_feat = self.word_embedding(previous_embedding_id).unsqueeze(0)

        attn_output, _ = self.attention(query=word_feat, 
                                        key=img_feat, 
                                        value=img_feat)
        combined = attn_output.squeeze(0) + word_feat.squeeze(0)
        output = self.llm_model(combined)
        return output


CrossEntropyLoss = nn.CrossEntropyLoss()
model1 = algorithem1()
optimizer = optim.Adam(model1.parameters(), lr= 0.0001)

transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor()
])


idx = {"idx1" : (0,1,2,3,4,5,6,7), "idx2" : (0,1,8,9,9,4,5,6,7), "idx3" : (0,1,10,11,4,5,6,7)}
count=0
for epoch in range(100):
    order = [1, 2, 3]
    random.shuffle(order)
    for count in (order):
        img = Image.open(f"C:/Users/jeonghyeon/Desktop/다운로드 ({count}).jpg").convert("RGB")
        img = transform(img).unsqueeze(0)
        final_token_index = torch.tensor([0], dtype=torch.long) # 만들때 테스트 한거 & 처음 시작 할때 입력값   
        for y in idx[f"idx{count}"]:
            optimizer.zero_grad()
            label = torch.tensor([y], dtype=torch.long)
            output1 = model1(img,final_token_index)# 각 단어의 확률
            
            index = output1.argmax().item()#가장 높은 확률의 단어의 인덱스 값
            max = output1.max().item()#가장 높은 확률
            actual_token = token_id_list[index]#가장 확률이 높은 단어의 토큰
            final_token_index = torch.tensor([y], dtype=torch.long)#가장 높은 확률의 단어의 인덱스 값을 파이트로치 형식으로 바꾼값
            loss = CrossEntropyLoss(output1, label)
            loss.backward()
            optimizer.step()
            print(f"정답: {idx_to_token[y]}  |  예측: {idx_to_token[index]} | loss:{loss}")
        torch.save(model1.state_dict(),"p1.pth")

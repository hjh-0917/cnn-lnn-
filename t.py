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
            nn.Flatten()
        )
        self.Linear_model = nn.Sequential(
            nn.Linear(64 * 120 * 120, 1500),
            nn.ReLU(),
            nn.Linear(1500,768),
            nn.ReLU()
        )
        
        self.word_embedding = nn.Embedding(word_count, 768)

        self.llm_model = nn.Sequential(
            nn.Linear(768 ,word_count )
        )


    def forward(self,img, previous_embedding_id):
        x1 = self.cnn_model(img)
        x2 = self.Linear_model(x1)
        word_feature = self.word_embedding(previous_embedding_id)
        combined = x2 + word_feature
        output = self.llm_model(combined)
        return output


CrossEntropyLoss = nn.CrossEntropyLoss()
model1 = algorithem1()
optimizer = optim.SGD(model1.parameters(), lr= 0.01)

transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor()
])


idx = {"idx1" : (1,2,3,4,5,6), "idx2" : (1,8,9,9,4,5,6), "idx3" : (1,10,11,4,5,6)}
count=0
for epoch in range(10):
    for count in range(1,4):
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
            final_token_index = torch.tensor([y], dtype=torch.long)#가장 높은 확률의 단어의 인덱스 값을 파이트로치 형식으로 바꾼거 그리고 나중에는 y말고 index넣어야함
            loss = CrossEntropyLoss(output1, label)
            loss.backward()
            optimizer.step()
            print(f"정답: {idx_to_token[y]}  |  예측: {idx_to_token[index]}")
        torch.save(model1.state_dict(),"p1.pth")

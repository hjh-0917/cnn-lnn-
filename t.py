import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import time
import os
import torchvision.transforms as transforms
import json
import random
import os
import pandas as pd

#graph 
import matplotlib.pyplot as plt

#JSON
with open('C:/Users/jeonghyeon/algorithem/Vocabulary.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
token_id_list = tuple(vocab.values())

with open('C:/Users/jeonghyeon/algorithem/revocabulary.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(data)

with open('idx.json', 'r', encoding='utf-8') as f:
    idx = json.load(f)

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

if os.path.exists("C:/Users/jeonghyeon/algorithem/p1.pth"):
    model1.load_state_dict(torch.load("C:/Users/jeonghyeon/algorithem/p1.pth"))
    print ("가중치 불러옴")
else:
    print("가중치 불러오기 실패")

df = pd.read_csv('C:/Users/jeonghyeon/algorithem/project/captions.txt')

loss_history = []  # loss 값 저장할 리스트

start = time.time()
for epoch in range(100):
    df = df.sample(frac=1).reset_index(drop=True)  ###############################
    for i, row in df.iterrows():
        img = Image.open(f"C:/Users/jeonghyeon/algorithem/project/Images/{row['image']}").convert("RGB")###  thank you for ai!!!!!!
        img = transform(img).unsqueeze(0)
        final_token_index = torch.tensor([0], dtype=torch.long)
        for y in idx[f"idx{i}"]:
            optimizer.zero_grad()
            label = torch.tensor([y], dtype=torch.long)
            output1 = model1(img,final_token_index)# 각 단어의 확률
            
            index = output1.argmax().item()#가장 높은 확률의 단어의 인덱스 값
            max = output1.max().item()#가장 높은 확률
            actual_token = token_id_list[index]#가장 확률이 높은 단어의 토큰
            final_token_index = torch.tensor([y], dtype=torch.long)#가장 높은 확률의 단어의 인덱스 값을 파이트로치 형식으로 바꾼값
            loss = CrossEntropyLoss(output1, label)
            loss.backward()

            loss_history.append(loss.item())  # loss 값 저장 graph

            optimizer.step()
            end = time.time()
            print(f"정답: {idx_to_token[y]}  |  예측: {idx_to_token[index]} | loss:{loss} | epoch : {epoch} | time : {end - start:.2f} | {i} image")
    torch.save(model1.state_dict(),"p1.pth")

# 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig('loss_graph.png')  # 이미지로 저장
    plt.close()  # 메모리 해제 (중요!)
    print(f"epoch {epoch} 그래프 저장 완료")

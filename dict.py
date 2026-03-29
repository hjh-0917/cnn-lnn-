# 수정: captions.txt 전체로 vocab 만들기
import pandas as pd
from transformers import AutoTokenizer
import json

df = pd.read_csv('C:/Users/jeonghyeon/algorithem/project/captions.txt')  # Flickr8k captions.txt
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

dict = {}
redict = {}

for caption in df['caption']:  # 모든 캡션 순회
    tokens = tokenizer.tokenize(caption)
    token_ids = tokenizer.encode(caption)
    
    dict["CLS"] = token_ids[0]
    redict[token_ids[0]] = "CLS"
    
    for i, token in enumerate(tokens):
        dict[token] = token_ids[i+1]
        redict[token_ids[i+1]] = token
    
    dict["SEP"] = token_ids[-1]
    redict[token_ids[-1]] = "SEP"

with open('Vocabulary.json', 'w', encoding='utf-8') as f:
    json.dump(dict, f, ensure_ascii=False, indent=4)
with open('revocabulary.json', 'w', encoding='utf-8') as f:
    json.dump(redict, f, ensure_ascii=False, indent=4)


    

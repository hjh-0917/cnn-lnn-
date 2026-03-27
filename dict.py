''' 글 데이터와 이미지 데이터는 따로 저장  이미지 100장 글 데이터 100장'''
dict = {}
redict = {}
from data import d
from transformers import AutoTokenizer
import json
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
for i in range(3):
    data = d[i]
    tokens = tokenizer.tokenize(data)
    token_ids = tokenizer.encode(data)
    len_token = (len(tokens))
    dict["CLS"]=token_ids[0]
    redict[token_ids[0]] = "CLS"
    for _ in range (len_token):
        o = _ + 1
        dict[tokens[_]]=token_ids[o]
        redict[token_ids[o]]=tokens[_]
    dict["SEP"]= token_ids[len_token + 1]
    redict[token_ids[len_token + 1]] = "SEP"
print (dict)
print(redict)

with open('Vocabulary.json', 'w', encoding='utf-8') as f:
    json.dump(dict,f, ensure_ascii=False, indent=4)
with open('revocabulary.json', 'w', encoding='utf-8') as f:
    json.dump(redict,f, ensure_ascii=False, indent=4)


    

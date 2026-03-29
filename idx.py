import json
from transformers import AutoTokenizer
import json
import pandas as pd

# Vocabulary.json 로드
with open('C:/Users/jeonghyeon/algorithem/Vocabulary.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# token_id → vocab 인덱스 매핑 (vocab의 순서 기반)
token_id_to_vocab_idx = {v: i for i, v in enumerate(vocab.values())}

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# 자동으로 idx 딕셔너리 생성
df = pd.read_csv('C:/Users/jeonghyeon/algorithem/project/captions.txt')

idx = {}
for i, row in df.iterrows():
    token_ids = tokenizer.encode(row['caption'])
    vocab_indices = tuple(token_id_to_vocab_idx[tid] for tid in token_ids if tid in token_id_to_vocab_idx)
    idx[f"idx{i}"] = vocab_indices


# idx 자동 생성 후 저장
with open('idx.json', 'w', encoding='utf-8') as f:
    # tuple은 JSON 저장 안되므로 list로 변환
    json.dump({k: list(v) for k, v in idx.items()}, f, ensure_ascii=False, indent=4)
    
# prepare_Chinese_dataset.py
from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)
dataset = load_dataset("swaption2009/20k-en-zh-translation-pinyin-hsk")

with open("data/train.zh", "w", encoding="utf-8") as f_zh, \
     open("data/train.en", "w", encoding="utf-8") as f_en:
    for item in dataset['train']:
        lines = item['text'].split('\n')
        english = mandarin = None
        
        for line in lines:
            if line.startswith('english:'):
                english = line.replace('english:', '').strip()
            elif line.startswith('mandarin:'):
                mandarin = line.replace('mandarin:', '').strip()
        
        if english and mandarin:
            f_zh.write(mandarin + "\n")
            f_en.write(english + "\n")

print("Done! Created data/train.zh and data/train.en")
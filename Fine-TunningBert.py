from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import os

notebook_login()
dataset = load_dataset("json", data_files={"train": "/content/train.jsonl", "test": "/content/validation.jsonl"})

print(dataset)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

mapDict = {
    "No Hate Speech": 0,
    "Hate Speech": 1
}

def transform_labels(labels):
  label = label['output']
  result = []
  for l in label:
    result.append(mapDict[l])
  return {"label": result}

def tokenize_function(example):
  return tokenizer(example['input'], padding=True, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.map(transform_labels, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

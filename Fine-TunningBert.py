from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import os

notebook_login()
dataset = load_dataset("json", data_files={"train": "/content/train.jsonl", "test": "/content/validation.jsonl"})

print(dataset)

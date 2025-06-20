from transformers import Trainer, AutoModelForSequenceclassification
from transformers import TrainingArguments
from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
import evaluate
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

output_dir = "./bert-hate-speech-test"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=200,
    save_total_limit=2,
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

model - AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

os.environ['WAND_DISABLE']="true"
os.environ['WANDB_MODE']="offline"

metric = evaluate.load("accuracy")

def compute_metric(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis = -1)
  return metric.compute(predictions=predictions, references=labels)

trainer.train()

trainer.evaluate()

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)

# 1. 读取处理后数据
data_dir = r"C:\\Users\\Administrator\\Desktop\\spider-topic\\model"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
val_df   = pd.read_csv(os.path.join(data_dir, "val.csv"))

# 2. 定义 Dataset
class StanceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts   = df["text"].tolist()
        self.targets = df["target"].tolist()
        self.labels  = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.targets[idx] + " [SEP] " + self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 3. 初始化分词器和模型
tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-chinese", 
    cache_dir=data_dir
)
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-chinese", 
    num_labels=3, 
    cache_dir=data_dir
)

# 4. 创建 Dataset 和 Trainer
train_dataset = StanceDataset(train_df, tokenizer)
val_dataset   = StanceDataset(val_df, tokenizer)
data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
    output_dir=os.path.join(data_dir, "outputs"),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    accuracy = (preds == labels).astype(float).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 5. 开始微调
trainer.train()

# 6. 保存最佳模型
trainer.save_model(os.path.join(data_dir, "best_model"))

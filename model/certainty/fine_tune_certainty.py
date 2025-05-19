import xml.etree.ElementTree as ET
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# —— 1. 配置文件路径 —— #
train_xml_path = r"C:\Users\Administrator\Desktop\model\code\chinese_train.xml"
eval_xml_path  = r"C:\Users\Administrator\Desktop\model\code\chinese_eval.xml"
output_dir     = r"C:\Users\Administrator\Desktop\model\code\certainty_model"
os.makedirs(output_dir, exist_ok=True)

# —— 2. 解析 XML 并构建 DataFrame —— #
def load_certainty_dataset(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for sent in root.findall(".//sentence"):
        raw_text = "".join(sent.itertext()).strip()
        if not raw_text:
            continue
        label_str = sent.get("certainty", "").lower()
        if label_str in ("certain", "uncertain"):
            label = 0 if label_str == "certain" else 1
            rows.append({"text": raw_text, "label": label})
    return pd.DataFrame(rows)


train_df = load_certainty_dataset(train_xml_path)
eval_df  = load_certainty_dataset(eval_xml_path)

# —— 3. 初始化分词器 —— #
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# —— 4. 自定义 Dataset —— #
class CertaintyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = CertaintyDataset(train_df, tokenizer)
eval_dataset  = CertaintyDataset(eval_df, tokenizer)

# —— 5. DataCollator —— #
data_collator = DataCollatorWithPadding(tokenizer)

# —— 6. 定义评价指标 —— #
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# —— 7. 构建模型与训练参数 —— #
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese", 
    num_labels=2
)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# —— 8. Trainer 实例化并训练 —— #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(output_dir)

print(f"✅ 模型已训练并保存在：{output_dir}")
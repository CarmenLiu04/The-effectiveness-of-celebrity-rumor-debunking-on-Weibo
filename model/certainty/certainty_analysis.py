import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# —— 配置 —— #
# 训练好模型的保存路径
model_dir = r"C:\Users\Administrator\Desktop\model\code\certainty_model"
# 待处理的 CSV 文件路径
input_csv = r"C:\Users\Administrator\Desktop\model\code\weibo_with_highscores.csv"
# 输出带预测结果的 CSV
output_csv = r"C:\Users\Administrator\Desktop\model\code\dataset_with_certainty.csv"
# 批量推理的 batch 大小
batch_size = 32

# —— 1. 加载模型与分词器 —— #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.to(device)
model.eval()

# 标签映射
id2label = {0: "certain", 1: "uncertain"}

# —— 2. 读取 CSV —— #
df = pd.read_csv(input_csv, encoding='utf-8-sig')

# 新增空列
df["predicted_certainty"] = None

# —— 3. 批量推理函数 —— #
def predict_certainty(texts):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    preds = torch.argmax(logits, dim=-1).cpu().tolist()
    return [id2label[p] for p in preds]

# —— 4. 对每个 batch 进行预测并填充 DataFrame —— #
contents = df["content"].astype(str).tolist()
for i in tqdm(range(0, len(contents), batch_size), desc="Predicting"):
    batch_texts = contents[i:i+batch_size]
    batch_preds = predict_certainty(batch_texts)
    df.loc[i:i+len(batch_preds)-1, "predicted_certainty"] = batch_preds

# —— 5. 保存结果 —— #
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"✅ 已完成不确定性预测，结果保存到：{output_csv}")
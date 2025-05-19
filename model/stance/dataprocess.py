import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 读取 & 初步清洗
df = pd.read_excel(r'C:\Users\Administrator\Desktop\spider-topic\model\raw_train_all_onecol.xlsb', engine='pyxlsb')
df = df[df['In Use'] == 1].reset_index(drop=True)
df = df.rename(columns={'Text': 'text', 'Target 1': 'target', 'Stance 1': 'stance'})
df = df.dropna(subset=['text', 'target', 'stance']).drop_duplicates(subset=['text', 'target'])
df['label'] = df['stance'].map({'支持': 0, '反对': 1, '中立': 2})

# 2. 划分训练/验证集
train_df, val_df = train_test_split(
    df[['text', 'target', 'label']],
    test_size=0.1,
    stratify=df['label'],
    random_state=42
)

# 3. 保存处理后的数据集
save_dir = r'C:\Users\Administrator\Desktop\spider-topic\model'
os.makedirs(save_dir, exist_ok=True)
train_df.to_csv(os.path.join(save_dir, 'train.csv'), index=False, encoding='utf-8-sig')
val_df.to_csv(os.path.join(save_dir, 'val.csv'), index=False, encoding='utf-8-sig')
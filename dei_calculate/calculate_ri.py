import os
import json

# TODO: 修改为你的 JSON 文件所在文件夹路径
folder_path = r'D:\spider-topic\data_count'
# TODO: 修改为之前存放 DI 值的 JSON 文件路径
di_values_path = r'D:\spider-topic\DI\di_values.json'
# TODO: 如果想覆盖原文件，就把 output_path 设成和 di_values_path 一样；
#      或者写到一个新文件：
output_path = di_values_path  # 或者 r'/path/to/your/json/folder/di_ri_values.json'

# 1. 读取已有的 DI 值
with open(di_values_path, 'r', encoding='utf-8') as f:
    di_values = json.load(f)

# 2. 计算 RI，并更新到同一个结构下：将值改为 [di, ri]
for fname in os.listdir(folder_path):
    if not fname.lower().endswith('.json'):
        continue
    key = os.path.splitext(fname)[0]
    # 只处理在 DI 文件中已有的项
    if key not in di_values:
        continue

    file_path = os.path.join(folder_path, fname)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f'❌ 无法读取 {fname}：{e}')
        continue

    support = data.get('支持', 0)
    oppose  = data.get('反对', 0)
    neutral = data.get('中立', 0)
    total_all = support + oppose + neutral

    # 计算 RI
    if total_all != 0:
        ri = neutral / total_all
    else:
        ri = None  # 或根据需求设为 0

    # 原来 di
    di = di_values[key]
    # 更新为 [di, ri]
    di_values[key] = [di, ri]

# 3. 写回 JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(di_values, f, ensure_ascii=False, indent=4)

print(f'✅ 已更新 DI & RI 值文件：{output_path}')
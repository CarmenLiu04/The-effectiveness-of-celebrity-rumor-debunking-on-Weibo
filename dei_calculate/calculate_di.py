import os
import json

# TODO: 修改为你的 JSON 文件所在文件夹路径
folder_path = r'D:\spider-topic\data_count'
# TODO: 修改为你希望输出的 DI 值文件路径
output_path = r'D:\spider-topic\DI\di_values.json'

di_values = {}

for fname in os.listdir(folder_path):
    if not fname.lower().endswith('.json'):
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
    total = support + oppose

    # 避免除以零
    if total != 0:
        di = (oppose - support) / total
    else:
        di = None  # 或者设为 0，看需求

    # 用文件名（不含扩展名）作为 key
    key = os.path.splitext(fname)[0]
    di_values[key] = di

# 将所有 DI 值写入一个新的 JSON 文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(di_values, f, ensure_ascii=False, indent=4)

print(f'✅ 已生成 DI 值文件：{output_path}')
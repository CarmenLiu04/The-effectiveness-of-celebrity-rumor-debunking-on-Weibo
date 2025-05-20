import os
import json

# TODO: 修改为之前保存了 DI 和 RI 值的 JSON 文件路径
di_ri_path = r'D:\spider-topic\DI\di_values.json'
# TODO: 如果想覆盖原文件，就把 output_path 设成和 di_ri_path 一样；
#      或者写到一个新文件：
output_path = di_ri_path  # 或者 r'/path/to/your/json/folder/di_ri_dei_values.json'

# 1. 读取已有的 DI & RI 值
with open(di_ri_path, 'r', encoding='utf-8') as f:
    di_ri = json.load(f)

# 2. 计算 DEI 并更新结构为 [di, ri, dei]
for key, values in di_ri.items():
    # 确保当前值是列表且至少包含 DI 和 RI
    if isinstance(values, list) and len(values) >= 2:
        di, ri = values[0], values[1]
        # 计算 DEI = DI / RI，避免除以零或空值
        if ri not in (0, None):
            dei = di / ri
        else:
            dei = None
        # 更新为包含第三个值的列表
        di_ri[key] = [di, ri, dei]

# 3. 写回 JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(di_ri, f, ensure_ascii=False, indent=4)

print(f'✅ 已生成包含 DI、RI 和 DEI 的文件：{output_path}')
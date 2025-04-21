import pandas as pd

# 读取CSV文件
file_path = 'output_sequences.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 需要检查的列
columns_to_check = ['uniprot_id', 'sequence', 'topt', 'ec', 'domain']

# 检查是否存在缺失值的行
missing_rows = df[df[columns_to_check].isnull().any(axis=1)]

if missing_rows.empty:
    print("在这些列中没有缺失值的行。")
else:
    print(f"在这些列中有{len(missing_rows)}行存在缺失值，具体如下：")
    print(missing_rows)

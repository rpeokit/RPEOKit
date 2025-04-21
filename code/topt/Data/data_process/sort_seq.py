import pandas as pd

# 读取CSV文件
df = pd.read_csv('../data_all_real.csv')

# 添加一个新列，计算sequence的长度
df['sequence_length'] = df['sequence'].apply(len)

# 按sequence_length列排序
sorted_df = df.sort_values(by='sequence_length')

# 删除sequence_length列
sorted_df = sorted_df.drop(columns=['sequence_length'])

# 保存为新的CSV文件
sorted_df.to_csv('../sorted_file.csv', index=False)

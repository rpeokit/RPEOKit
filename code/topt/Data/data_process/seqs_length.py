import pandas as pd

# 读取CSV文件
file_path = '../Data/processed_data.csv'  # 替换为你的CSV文件路径
output_file_path = 'seqs10_300.csv'  # 替换为输出文件路径

# 设置需要筛选的序列长度区间（min_length 到 max_length）
min_length = 10  # 替换为你想要的最小长度
max_length = 300  # 替换为你想要的最大长度

# 读取CSV文件
df = pd.read_csv(file_path)

# 过滤出 'sequence' 列长度在指定区间的行
filtered_df = df[df['sequence'].apply(lambda x: min_length <= len(x) <= max_length)]

# 将过滤后的数据保存到新的CSV文件
filtered_df.to_csv(output_file_path, index=False)

# 输出生成的行数
print(f"生成了{len(filtered_df)}行满足条件的记录，并保存到'{output_file_path}'。")

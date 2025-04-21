import pandas as pd

# 读取CSV文件
df = pd.read_csv('../data_all_real.csv')

# 合并ec1, ec2, ec3, ec4列为ec_number列
df['ec_number'] = df['ec1'].astype(str) + '.' + df['ec2'].astype(str) + '.' + df['ec3'].astype(str) + '.' + df['ec4'].astype(str)

# 调整列的顺序，将ec_number放在domain列之前
columns = df.columns.tolist()
domain_columns = ['domain_Archaea', 'domain_Bacteria', 'domain_Eukarya']
# 找到domain列的位置
for col in domain_columns:
    if col in columns:
        columns.remove(col)  # 移除原有位置
for col in domain_columns:
    columns.insert(columns.index('ec_number') + 1, col)  # 在ec_number后插入

# 重新排列列
df = df[columns]
df.drop(['ec1', 'ec2', 'ec3', 'ec4'], axis=1, inplace=True)
# 保存到新的CSV文件
df.to_csv('output_file.csv', index=False)

print("合并完成，ec_number已放在domain列之前，结果已保存到output_file.csv")

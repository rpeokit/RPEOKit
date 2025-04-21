import pandas as pd

# 读取CSV文件
csv_file = '../output_sequences.csv'  # 替换为您的CSV文件名
data = pd.read_csv(csv_file)

# 拆分EC号并数值化
# 将EC号拆分为三列并转换为数值
ec_split = data['ec'].str.split('.', expand=True)
ec_split.columns = ['ec1', 'ec2', 'ec3', 'ec4']
data['ec1'] = ec_split['ec1'].astype(int)
data['ec2'] = ec_split['ec2'].astype(int)
data['ec3'] = ec_split['ec3'].astype(int)
data['ec4'] = ec_split['ec4'].astype(int)

# 对Domain进行one-hot编码
domain_one_hot = pd.get_dummies(data['domain'], prefix='domain').astype(int)
data = pd.concat([data, domain_one_hot], axis=1)

# 删除原始的ec和domain列
data.drop(['ec', 'domain'], axis=1, inplace=True)

# 保存处理后的数据到新的CSV文件
output_file = '../data_all_real.csv'  # 替换为您想要保存的文件名
data.to_csv(output_file, index=False)

print("数据处理完成，已保存到", output_file)

import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
csv_file = '../train_1.csv'  # 你要处理的文件名
df = pd.read_csv(csv_file)

# 随机抽取 20% 数据作为测试集，剩下 80% 作为训练集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 将抽取的数据保存到新的文件中
train_df.to_csv('../train_data.csv', index=False)  # 保存训练数据
test_df.to_csv('../test_data.csv', index=False)    # 保存测试数据

print("Data has been split and saved.")

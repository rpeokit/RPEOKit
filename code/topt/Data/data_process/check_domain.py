import pandas as pd

# 读取CSV文件
file_path = 'output_sequences.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 检查'domain'这一列是否存在
if 'domain' in df.columns:
    # 获取'domain'列中的唯一值
    unique_domains = df['domain'].unique()
    
    # 输出结果
    print(f"'domain'列中共有{len(unique_domains)}种不同的元素，分别是：")
    for domain in unique_domains:
        print(domain)
else:
    print("CSV文件中没有'domain'这一列，请检查文件格式。")

import pandas as pd

def find_max_in_fourth_last_column(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 获取倒数第四列
    fourth_last_column = df.iloc[:, -4]
    
    # 查找倒数第四列的最大值
    max_value = fourth_last_column.max()
    
    return max_value

# 示例用法
csv_file = '../data_all_real.csv'  # 将文件名替换为实际的CSV文件名
max_value = find_max_in_fourth_last_column(csv_file)
print(f"倒数第四列的最大值是: {max_value}")

import pandas as pd

# 读取原始CSV文件
input_file = '../Data/data_all_real.csv'  # 请替换为你的CSV文件名
output_file = '../Data/id_domain.csv'  # 输出文件名

# 加载数据
data = pd.read_csv(input_file)

# 提取ID和EC列
extracted_data = data[['uniprot_id', 'domain_Archaea', 'domain_Bacteria', 'domain_Eukarya']]

# 保存到新的CSV文件
extracted_data.to_csv(output_file, index=False)

print(f"提取完成，数据已保存到 {output_file}")

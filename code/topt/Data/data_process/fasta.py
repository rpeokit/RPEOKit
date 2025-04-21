import pandas as pd

# 读取CSV文件
csv_file_path = '../sorted_file.csv'  # 替换为你的CSV文件路径
output_fasta_path = '../fasta/output.fasta'  # 输出FASTA文件路径

# 加载CSV文件
df = pd.read_csv(csv_file_path)

# 提取uniprot_id和sequence列
uniprot_ids = df['uniprot_id']
sequences = df['sequence']

# 写入FASTA文件
with open(output_fasta_path, 'w') as fasta_file:
    for uniprot_id, sequence in zip(uniprot_ids, sequences):
        fasta_file.write(f'>{uniprot_id}\n')  # 写入ID行
        fasta_file.write(f'{sequence}\n')     # 写入序列行

print(f"FASTA file has been created at: {output_fasta_path}")

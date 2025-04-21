import pandas as pd

# 读取TSV文件
def load_tsv(tsv_file):
    # 读取TSV文件
    df = pd.read_csv(tsv_file, sep='\t', header=None)
    # 提取id列
    tsv_ids = df[1].tolist()  # 假设ID在第二列
    return set(tsv_ids)  # 使用集合以便于后续对比

# 读取FASTA文件
def load_fasta(fasta_file):
    fasta_ids = set()
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # 获取uniprot_id
                fasta_id = line.split('|')[0][1:].strip()  # 去掉`>`并去掉空格
                fasta_ids.add(fasta_id)
    return fasta_ids

# 对比ID
def compare_ids(tsv_ids, fasta_ids):
    unmatched_tsv_ids = tsv_ids - fasta_ids
    unmatched_fasta_ids = fasta_ids - tsv_ids

    if unmatched_tsv_ids:
        print("TSV文件中未匹配的ID:")
        for id in unmatched_tsv_ids:
            print(id)
    else:
        print("TSV文件中的所有ID在FASTA文件中都有匹配。")

    if unmatched_fasta_ids:
        print("FASTA文件中未匹配的ID:")
        for id in unmatched_fasta_ids:
            print(id)
    else:
        print("FASTA文件中的所有ID在TSV文件中都有匹配。")

# 主程序
if __name__ == "__main__":
    tsv_file = './data/info.tsv'  # 替换为你的TSV文件路径
    fasta_file = './data/seqs_ec_topt.fasta'  # 替换为你的FASTA文件路径

    # 加载ID
    tsv_ids = load_tsv(tsv_file)
    fasta_ids = load_fasta(fasta_file)

    # 对比ID
    compare_ids(tsv_ids, fasta_ids)

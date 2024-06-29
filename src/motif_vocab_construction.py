import multiprocessing as mp
import os
import os.path as path
import pickle
from collections import Counter
from datetime import datetime
from functools import partial
from typing import List, Tuple
import shutil
import random
from tqdm import tqdm

from arguments import parse_arguments
from model.mol_graph import MolGraph
from model.mydataclass import Paths


# Frequency-based method:
def apply_frequency_based_operations(batch: List[Tuple[int, str]], mols_pkl_dir: str) -> Counter:
    vocab = Counter()
    pos = mp.current_process()._identity[0]
    with tqdm(total=len(batch), desc=f"Processing {pos}", position=pos - 1, ncols=80, leave=False) as pbar:
        for idx, smi in batch:
            # 实现应用“频繁合并操作集”完成merging构造与motif分割
            mol = MolGraph(smi, tokenizer="motif", methods="frequency_based")
            # 将存储此时的MolGraph对象存储为二进制文件，
            with open(path.join(mols_pkl_dir, f"{idx}.pkl"), "wb") as f:
                pickle.dump(mol, f)
            # 单独提炼motif列表，逐步搭建vocab列表
            vocab = vocab + Counter(mol.motifs)
            pbar.update()
    return vocab

def frequency_based_motif_vocab_construction(
        train_path: str,
        vocab_path: str,
        operation_path: str,
        num_operations: int,
        num_workers: int,
        mols_pkl_dir: str,
):
    print(f"[{datetime.now()}] Constructing motif vocabulary from {train_path}.")
    print(f"Number of workers: {num_workers}. Total number of CPUs: {mp.cpu_count()}.")

    # 1. 加载训练数据：batches：[第一个batch,...,第n个batch] --- 第一个batch： [(id_1,SMILES_1),...,(id_batch_size,SMILES_batch_size)]
    data_set = [(idx, smi.strip("\n")) for idx, smi in enumerate(open(train_path))]
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i: i + batch_size] for i in range(0, len(data_set), batch_size)]
    print(f"Total: {len(data_set)} molecules.\n")

    print(f"Processing...")
    vocab = Counter()
    os.makedirs(mols_pkl_dir, exist_ok=True)
    # 2. 加载操作集
    MolGraph.load_operations(operation_path, num_operations)

    # 3. 应用操作集
    func = partial(apply_frequency_based_operations, mols_pkl_dir=mols_pkl_dir)
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        for batch_vocab in pool.imap(func, batches):
            vocab = vocab + batch_vocab

    # 得到完整的vocab词典——每个键是一个元组： （【无断开点信息的motif片段的smiles表示['motif_no_conn']】，【处理完断开点后，有连接信息（子图方法）的motif片段的smiles表示 ['motif']】）

    # 存储
    # 提取 vocab 中不在 MolGraph.OPERATIONS 列表中的MOTIF（即原子）
    atom_list = [x for (x, _) in vocab.keys() if x not in MolGraph.OPERATIONS]
    atom_list.sort()
    new_vocab = []
    full_list = atom_list + MolGraph.OPERATIONS

    for (x, y), value in vocab.items():
        assert x in full_list
        new_vocab.append((x, y, value))

    # 重构顺序：按照operation次序(即motif在训练集中的频率)排序
    index_dict = dict(zip(full_list, range(len(full_list))))
    sorted_vocab = sorted(new_vocab, key=lambda x: index_dict[x[0]])

    with open(vocab_path, "w") as f: # self.vocab_path = path.join(self.preprocess_dir, "vocab.txt")
        for (x, y, _) in sorted_vocab:
            f.write(f"{x} {y}\n")

    print(f"\r[{datetime.now()}] Motif vocabulary construction finished.")
    print(f"The motif vocabulary is in {vocab_path}.\n\n")

# Connectivity-based method:
def apply_connectivity_based_operations(batch: List[Tuple[int, str]], mols_pkl_dir: str) -> Counter:
    vocab = Counter()
    pos = mp.current_process()._identity[0]
    with tqdm(total=len(batch), desc=f"Processing {pos}", position=pos - 1, ncols=80, leave=False) as pbar:
        for idx, smi in batch:
            # 实现应用“连通性”完成merging构造与motif分割
            mol = MolGraph(smi, tokenizer="motif", methods="connectivity_based")
            # 将存储此时的MolGraph对象存储为二进制文件，
            with open(path.join(mols_pkl_dir, f"{idx}.pkl"), "wb") as f:
                pickle.dump(mol, f)
            # 单独提炼motif列表，逐步搭建vocab列表
            vocab = vocab + Counter(mol.motifs)
            pbar.update()
    return vocab

def connectivity_based_motif_vocab_construction(
        train_path: str,
        vocab_path: str,
        num_workers: int,
        mols_pkl_dir: str,
):
    print(f"[{datetime.now()}] Constructing motif vocabulary from {train_path}.")
    print(f"Number of workers: {num_workers}. Total number of CPUs: {mp.cpu_count()}.")

    # 1. 加载训练数据
    data_set = [(idx, smi.strip("\n")) for idx, smi in enumerate(open(train_path))]
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i: i + batch_size] for i in range(0, len(data_set), batch_size)]
    print(f"Total: {len(data_set)} molecules.\n")

    print(f"Processing...")
    vocab = Counter()
    os.makedirs(mols_pkl_dir, exist_ok=True)

    # 2. 应用操作集
    func = partial(apply_connectivity_based_operations, mols_pkl_dir=mols_pkl_dir)
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        for batch_vocab in pool.imap(func, batches):
            vocab = vocab + batch_vocab

    # get_merging_graph_by_connectivity中已经考虑了叶节点的断开，也就是已经有原子了
    # atom_list = [x for (x, _) in vocab.keys()]
    # atom_list.sort()

    # 得到完整的vocab词典
    new_vocab = []
    for (x, y), value in vocab.items():
        new_vocab.append((x, y, value))

    # 重构顺序
    # 按照motif在训练集中的【出现频率】排序
    sorted_vocab = sorted(new_vocab, key=lambda x: x[2])

    with open(vocab_path, "w") as f:
        for (x, y, _) in sorted_vocab:
            f.write(f"{x} {y}\n")

    print(f"\r[{datetime.now()}] Motif vocabulary construction finished.")
    print(f"The motif vocabulary is in {vocab_path}.\n\n")


# 词典的【集成函数】
def vocab_ensemble(freq_vocab_path: str, conn_vocab_path: str, vocab_path: str):
    """
    合并频率词典和连通性词典，去重后保存到最终词典路径。
    :param freq_vocab_path: 频率词典路径
    :param conn_vocab_path: 连通性词典路径
    :param vocab_path: 最终词典路径
    """
    # 读取频率词典
    with open(freq_vocab_path, "r") as f:
        freq_vocab = [line.strip() for line in f.readlines()]

    # 读取连通性词典
    with open(conn_vocab_path, "r") as f:
        conn_vocab = [line.strip() for line in f.readlines()]

    # 将词典条目存储到集合中以去重
    freq_vocab_set = set(freq_vocab)
    conn_vocab_set = set(conn_vocab)

    # 合并词典条目，保留频率词典中已有的词条
    ensemble_vocab = list(freq_vocab_set)
    for motif in conn_vocab_set:
        if motif not in freq_vocab_set:
            ensemble_vocab.append(motif)

    # 保存合并后的词典
    with open(vocab_path, "w") as f:
        for entry in ensemble_vocab:
            f.write(entry + "\n")

    print(f"Vocabulary merged and saved to {vocab_path}.")



# MolGraph数据的【集成函数】：针对method == 'ensemble'。
def copy_file_wrapper(args):
    return copy_file(*args)

def copy_file(src_file, dest_file):
    shutil.copy(src_file, dest_file)
    return 1

def ensemble_mol_graphs(freq_mols_pkl_dir: str, conn_mols_pkl_dir: str, mols_pkl_dir: str, data_ensemble_mode: str = 'random', num_workers: int = mp.cpu_count(), random_rate: float = 0.5):
    """
    集成频率和连通性MolGraph数据到工作区
    :param freq_mols_pkl_dir: 依据频率构建的molgraph对象的pkl文件路径
    :param conn_mols_pkl_dir: 依据连通性构建的molgraph对象的pkl文件路径
    :param mols_pkl_dir: 集成后的molgraph对象的pkl文件路径
    :param data_ensemble_mode: 数据集成模式，"overlay"或"random"
    :param num_workers: 使用的线程数
    """
    # 清空工作区mols_pkl_dir的现有文件
    if path.exists(mols_pkl_dir):
        shutil.rmtree(mols_pkl_dir)
    os.makedirs(mols_pkl_dir, exist_ok=True)

    # 检查源目录是否不为空且数据量相同
    freq_files = sorted(os.listdir(freq_mols_pkl_dir))
    conn_files = sorted(os.listdir(conn_mols_pkl_dir))

    if not freq_files or not conn_files:
        raise ValueError("Source directories must not be empty.")
    if len(freq_files) != len(conn_files):
        raise ValueError("Source directories must have the same number of files.")

    tasks = []
    if data_ensemble_mode == "overlay":
        # 数据集成的叠加模式
        # 复制freq_mols_pkl_dir中的文件到工作区
        for file in freq_files:
            tasks.append((path.join(freq_mols_pkl_dir, file), path.join(mols_pkl_dir, file)))

        # 复制conn_mols_pkl_dir中的文件到工作区，文件名依序改名
        offset = len(freq_files)
        for i, file in enumerate(conn_files):
            new_name = f"{i + offset}.pkl"
            tasks.append((path.join(conn_mols_pkl_dir, file), path.join(mols_pkl_dir, new_name)))

    elif data_ensemble_mode == "random":
        # 数据集成的随机模式
        for freq_file, conn_file in zip(freq_files, conn_files):
            if random.random() < random_rate:
                tasks.append((path.join(freq_mols_pkl_dir, freq_file), path.join(mols_pkl_dir, freq_file)))
            else:
                tasks.append((path.join(conn_mols_pkl_dir, conn_file), path.join(mols_pkl_dir, freq_file)))

    else:
        raise ValueError("Invalid data_ensemble_mode. Choose 'overlay' or 'random'.")

    # 多线程处理文件复制
    with mp.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(copy_file_wrapper, tasks), total=len(tasks), desc="Integrating MolGraph data"))

    print(f"MolGraph data integrated and saved to {mols_pkl_dir}.")



if __name__ == "__main__":
    args = parse_arguments()
    paths = Paths(args)
    os.makedirs(paths.preprocess_dir, exist_ok=True)

    if args.method == 'ensemble':
        # frequency_based_motif_vocab_construction(
        #     train_path=paths.train_path,
        #     vocab_path=paths.freq_vocab_path, # vocab constructed by frequency_based_motif_vocab_construction()
        #     operation_path=paths.operation_path,
        #     num_operations=args.num_operations,
        #     mols_pkl_dir=paths.freq_mols_pkl_dir, # Train.smiles -> MolGraph(.pkl) set by frequency_based_motif_vocab_construction()
        #     num_workers=args.num_workers,
        # )
        # connectivity_based_motif_vocab_construction(
        #     train_path=paths.train_path,
        #     vocab_path=paths.conn_vocab_path, # vocab constructed by connectivity_based_motif_vocab_construction()
        #     mols_pkl_dir=paths.conn_mols_pkl_dir, # Train.smiles -> MolGraph(.pkl) set by connectivity_based_motif_vocab_construction()
        #     num_workers=args.num_workers,
        # )
        # # 调用合并函数，将freq_vocab_path中的词典和conn_vocab_path中的词典合并，合并结果【引入工作区】——存储到paths.vocab_path
        # vocab_ensemble(
        #     freq_vocab_path=paths.freq_vocab_path ,
        #     conn_vocab_path=paths.conn_vocab_path ,
        #     vocab_path=paths.vocab_path,
        # )
        # 调用MolGraph数据的【集成函数】，将freq_mols_pkl_dir和conn_mols_pkl_dir中的文件集成后放入工作区
        ensemble_mol_graphs(
            freq_mols_pkl_dir=paths.freq_mols_pkl_dir,
            conn_mols_pkl_dir=paths.conn_mols_pkl_dir,
            mols_pkl_dir=paths.mols_pkl_dir,
            data_ensemble_mode=args.data_ensemble_mode,
            random_rate=0.9,
        )
    elif args.method == 'frequency_based_only': # 不同method对应的数据会分至不同文件夹保存，无需建立工作区
        frequency_based_motif_vocab_construction(
            train_path=paths.train_path,
            vocab_path=paths.vocab_path, # vocab constructed by frequency_based_motif_vocab_construction()
            operation_path=paths.operation_path,
            num_operations=args.num_operations,
            mols_pkl_dir=paths.mols_pkl_dir, # Train.smiles -> MolGraph(.pkl) set by frequency_based_motif_vocab_construction()
            num_workers=args.num_workers,
        )
        
    elif args.method == 'connectivity_based_only': # 不同method对应的数据会分至不同文件夹保存，无需建立工作区
        connectivity_based_motif_vocab_construction(
            train_path=paths.train_path,
            vocab_path=paths.vocab_path, # vocab constructed by connectivity_based_motif_vocab_construction()
            mols_pkl_dir=paths.mols_pkl_dir, # Train.smiles -> MolGraph(.pkl) set by connectivity_based_motif_vocab_construction()
            num_workers=args.num_workers,
        )



'''
# 词典的【引入工作区函数】
def vocab_import(source_vocab_path: str, target_vocab_path: str):
    """
    将源词典复制到目标路径——目标路径的词典被命名为vocab.txt。
    :param source_vocab_path: 源词典路径
    :param target_vocab_path: 目标词典路径
    """
    import shutil
    shutil.copy(source_vocab_path, target_vocab_path)
    print(f"Vocabulary imported from {source_vocab_path} to {target_vocab_path}.")
'''
    
'''
# MolGraph数据的【引入工作区函数】：针对method == 'connectivity_based_only'和method == 'frequency_based_only'。
def import_mol_graphs(source_mols_pkl_dir: str, target_mols_pkl_dir: str):
    """
    将源MolGraph数据导入到工作区
    :param source_mols_pkl_dir: 源MolGraph数据路径
    :param target_mols_pkl_dir: 目标工作区路径
    """
    # 清空工作区target_mols_pkl_dir的现有文件
    if path.exists(target_mols_pkl_dir):
        shutil.rmtree(target_mols_pkl_dir)
    os.makedirs(target_mols_pkl_dir, exist_ok=True)

    # 检查源目录是否不为空
    source_files = sorted(os.listdir(source_mols_pkl_dir))
    if not source_files:
        raise ValueError("Source directory must not be empty.")

    # 复制源目录中的文件到工作区
    for file in source_files:
        shutil.copy(path.join(source_mols_pkl_dir, file), path.join(target_mols_pkl_dir, file))

    print(f"MolGraph data imported from {source_mols_pkl_dir} to {target_mols_pkl_dir}.")
'''
import multiprocessing as mp
import os
import os.path as path
import pickle
from collections import Counter
from datetime import datetime
from functools import partial
from typing import List, Tuple, Dict, NamedTuple, Optional

from tqdm import tqdm
from rdkit import Chem

from arguments import parse_arguments
from model.mol_graph import MolGraph
from model.mydataclass import Paths
from model.utils import smiles2mol, fragment2smiles

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

    # 重构顺序→对后续操作可能有影响？
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

    # 得到完整的vocab词典
    atom_list = [x for (x, _) in vocab.keys()]
    atom_list.sort()
    new_vocab = []

    for (x, y), value in vocab.items():
        new_vocab.append((x, y, value))

    # 重构顺序
    sorted_vocab = sorted(new_vocab, key=lambda x: x[0])

    with open(vocab_path, "w") as f:
        for (x, y, _) in sorted_vocab:
            f.write(f"{x} {y}\n")

    print(f"\r[{datetime.now()}] Motif vocabulary construction finished.")
    print(f"The motif vocabulary is in {vocab_path}.\n\n")


if __name__ == "__main__":
    args = parse_arguments()
    paths = Paths(args)
    os.makedirs(paths.preprocess_dir, exist_ok=True)

    if args.method == 'frequency_based':
        frequency_based_motif_vocab_construction(
            train_path=paths.train_path,
            vocab_path=paths.vocab_path,
            operation_path=paths.operation_path,
            num_operations=args.num_operations,
            mols_pkl_dir=paths.mols_pkl_dir,
            num_workers=args.num_workers,
        )
    elif args.method == 'connectivity_based':
        connectivity_based_motif_vocab_construction(
            train_path=paths.train_path,
            vocab_path=paths.vocab_path,
            mols_pkl_dir=paths.mols_pkl_dir,
            num_workers=args.num_workers,
        )

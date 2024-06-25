import multiprocessing as mp
import os
import os.path as path
import pickle
from datetime import datetime
from functools import partial
from typing import List, Tuple

import torch
from tqdm import tqdm

from arguments import parse_arguments
from model.mol_graph import MolGraph
from model.mydataclass import Paths


def process_train_batch(batch: List[str], raw_dir: str, save_dir: str):
    pos = mp.current_process()._identity[0]
    with tqdm(total = len(batch), desc=f"Processing {pos}", position=pos-1, ncols=80, leave=False) as pbar:
        for file in batch:
            with open(path.join(raw_dir, file), "rb") as f:
                mol: MolGraph = pickle.load(f) # 训练数据在motif_vocab构建过程中就已经完成了【MolGraph对象化处理】：已经将每一行smiles对应的分子构建为MolGraph对象，并保存为了二进制文件
            data = mol.get_data() # 调用get_data：为batch中的每个【MolGraph对象】构建训练数据，训练数据中的【“查询拼接过程”子图data对象】表现了生成过程
            torch.save(data, path.join(save_dir, file.split()[0]+".pth"))# 用【原始文件名{idx}.pkl去掉pkl后的idx】作为文件保存
            pbar.update()

def process_valid_batch(batch: List[Tuple[int, str]], save_dir: str):
    pos = mp.current_process()._identity[0]
    with tqdm(total = len(batch), desc=f"Processing {pos}", position=pos-1, ncols=80, leave=False) as pbar:
        for idx, smi in batch:
            mol = MolGraph(smi, tokenizer="motif", methods = ?) # 测试数据需要完成【MolGraph对象化处理】，调用MolGraph的过程涉及motif拆分方法（merging_graph构建方法）的选择

            data = mol.get_data() # 调用get_data：为batch中的每个【MolGraph对象】构建训练数据，训练数据中的【“查询拼接过程”子图data对象】表现了生成过程
            torch.save(data, path.join(save_dir, f"{idx}.pth")) # 用【该分子在valid.smiles文件中的标号】作为文件名保存
            pbar.update()

def make_trainig_data(
    mols_pkl_dir: str,
    valid_path: str,
    vocab_path: str,
    train_processed_dir: str,
    valid_processed_dir: str,
    vocab_processed_path: str,
    num_workers: int,
):

    print(f"[{datetime.now()}] Preprocessing traing data.")
    print(f"Number of workers: {num_workers}. Total number of CPUs: {mp.cpu_count()}.\n")

# 1. 生成训练数据：数据预处理+自回归数据标注
    print(f"[{datetime.now()}] Loading training set from {mols_pkl_dir}.\n")
    os.makedirs(train_processed_dir, exist_ok=True)
    # 训练数据集：训练数据在motif_vocab构建过程中就已经完成了【列表化加载】，并且对列表完成了【MolGraph对象化处理】：已经将每一行smiles对应的分子构建为MolGraph对象，并保存为了二进制文件
    data_set = os.listdir(mols_pkl_dir) # 使用 os.listdir 列出原始数据目录 mols_pkl_dir 中的所有文件名，得到 data_set。
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i : i + batch_size] for i in range(0, len(data_set), batch_size)]
    # 多线程调用process_train_batch处理mols_pkl_dir
    func = partial(process_train_batch, raw_dir=mols_pkl_dir, save_dir=train_processed_dir)
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        pool.map(func, batches)

# 2. 生成测试数据：数据预处理+自回归数据标注
    print(f"[{datetime.now()}] Preprocessing valid set from {valid_path}.\n")
    os.makedirs(valid_processed_dir, exist_ok=True)
    # 测试数据集：测试数据集导入原始的valid.smiles以完成【列表化加载】
    data_set = [(idx, smi.strip("\n")) for idx, smi in enumerate(open(valid_path))] #【smiles文件→[第一个barch[(id_1,SMILES_1),...,(id_batch_size,SMILES_batch_size)],...,第n个batch]】
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i : i + batch_size] for i in range(0, len(data_set), batch_size)]
    # 多线程调用process_valid_batch处理mols_pkl_dir
    func = partial(process_valid_batch, save_dir=valid_processed_dir)
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        pool.map(func, batches)

# 3. 为训练专门处理Motif词典
    print(f"[{datetime.now()}] Preprocessing motif vocabulary from {vocab_path}.\n")
    vocab_data = MolGraph.preprocess_vocab()
    with open(vocab_processed_path, "wb") as f:
        torch.save(vocab_data, f) # 保存到 vocab.pth

    print(f"[{datetime.now()}] Preprocessing finished.\n\n")

if __name__ == "__main__":

    args = parse_arguments()
    paths  = Paths(args)

    MolGraph.load_operations(paths.operation_path, args.num_operations) # 导入操作集

    MolGraph.load_vocab(paths.vocab_path) # 调用类方法load_vocab()

    make_trainig_data(
        mols_pkl_dir = paths.mols_pkl_dir, # 构造训练数据的时候需要用到mols存储对象！
        valid_path = paths.valid_path,
        vocab_path = paths.vocab_path,
        train_processed_dir = paths.train_processed_dir,
        valid_processed_dir = paths.valid_processed_dir,
        vocab_processed_path = paths.vocab_processed_path,
        num_workers = args.num_workers,
    )

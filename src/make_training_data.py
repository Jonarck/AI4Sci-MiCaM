import multiprocessing as mp
import os
import os.path as path
import pickle
from datetime import datetime
from functools import partial
from typing import List, Tuple
import shutil
import random
import torch
from tqdm import tqdm

from arguments import parse_arguments
from model.mol_graph import MolGraph
from model.mydataclass import Paths


def process_train_batch(batch: List[str], raw_dir: str, save_dir: str):
    pos = mp.current_process()._identity[0]
    with tqdm(total=len(batch), desc=f"Processing {pos}", position=pos-1, ncols=80, leave=False) as pbar:
        for file in batch:
            try:
                with open(path.join(raw_dir, file), "rb") as f:
                    mol: MolGraph = pickle.load(f)
                data = mol.get_data()
                torch.save(data, path.join(save_dir, file.split()[0] + ".pth"))
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error reading file {file}: {e}")
            pbar.update()


def process_valid_batch(batch: List[Tuple[int, str]], save_dir: str, method: str = 'ensemble', data_ensemble_mode: str = 'random'):
    """
    处理valid数据的MolGraph对象化并保存到指定目录
    :param batch: 包含SMILES字符串的batch
    :param save_dir: 保存MolGraph对象的目录
    :param method: Vocab构建的方法选择, 默认是ensemble
    :param data_ensemble_mode: 数据集成模式, 默认是random
    """
    # 清空save_dir里的所有文件
    # if path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    # os.makedirs(save_dir, exist_ok=True)

    pos = mp.current_process()._identity[0]
    with tqdm(total=len(batch), desc=f"Processing {pos}", position=pos-1, ncols=80, leave=False) as pbar:
        offset = len(batch)
        for idx, smi in batch:
            if method == 'frequency_based_only':
                mol = MolGraph(smi, tokenizer="motif", methods="frequency_based")
                data = mol.get_data()
                torch.save(data, path.join(save_dir, f"{idx}.pth"))
            elif method == 'connectivity_based_only':
                mol = MolGraph(smi, tokenizer="motif", methods="connectivity_based")
                data = mol.get_data()
                torch.save(data, path.join(save_dir, f"{idx}.pth"))
            elif method == 'ensemble':
                if data_ensemble_mode == 'overlay':
                    # 处理 frequency_based 方法
                    mol_freq = MolGraph(smi, tokenizer="motif", methods="frequency_based")
                    data_freq = mol_freq.get_data()
                    torch.save(data_freq, path.join(save_dir, f"{idx}.pth"))

                    # 处理 connectivity_based 方法
                    mol_conn = MolGraph(smi, tokenizer="motif", methods="connectivity_based")
                    data_conn = mol_conn.get_data()
                    torch.save(data_conn, path.join(save_dir, f"{idx + offset}.pth"))
                elif data_ensemble_mode == 'random':
                    if random.random() < 0.9:
                        mol = MolGraph(smi, tokenizer="motif", methods="frequency_based")
                    else:
                        mol = MolGraph(smi, tokenizer="motif", methods="connectivity_based")
                    data = mol.get_data()
                    torch.save(data, path.join(save_dir, f"{idx}.pth"))
                else:
                    raise ValueError("Invalid data_ensemble_mode. Choose 'overlay' or 'random'.")
            else:
                raise ValueError("Invalid methods. Choose 'frequency_based_only', 'connectivity_based_only' or 'ensemble'.")
            pbar.update()

def make_trainig_data(
    mols_pkl_dir: str,
    valid_path: str,
    vocab_path: str,
    train_processed_dir: str,
    valid_processed_dir: str,
    vocab_processed_path: str,
    num_workers: int,
    methods,
):

    print(f"[{datetime.now()}] Preprocessing traing data.")
    print(f"Number of workers: {num_workers}. Total number of CPUs: {mp.cpu_count()}.\n")


    print(f"[{datetime.now()}] Loading training set from {mols_pkl_dir}.\n")
    os.makedirs(train_processed_dir, exist_ok=True)
    data_set = os.listdir(mols_pkl_dir)
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i : i + batch_size] for i in range(0, len(data_set), batch_size)]
    func = partial(process_train_batch, raw_dir=mols_pkl_dir, save_dir=train_processed_dir)
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        pool.map(func, batches)
    
    
    print(f"[{datetime.now()}] Preprocessing valid set from {valid_path}.\n")
    # 清空valid_processed_dir里的所有文件
    if path.exists(valid_processed_dir):
        shutil.rmtree(valid_processed_dir)
    os.makedirs(valid_processed_dir, exist_ok=True)
    data_set = [(idx, smi.strip("\n")) for idx, smi in enumerate(open(valid_path))]
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i : i + batch_size] for i in range(0, len(data_set), batch_size)]
    # edit valid form by changing "methods" and "data_ensemble_mode"
    func = partial(process_valid_batch, save_dir=valid_processed_dir, method=methods, data_ensemble_mode='random')
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        pool.map(func, batches)

    print(f"[{datetime.now()}] Preprocessing motif vocabulary from {vocab_path}.\n")
    vocab_data = MolGraph.preprocess_vocab()
    with open(vocab_processed_path, "wb") as f:
        torch.save(vocab_data, f)

    print(f"[{datetime.now()}] Preprocessing finished.\n\n")

if __name__ == "__main__":

    args = parse_arguments()
    paths  = Paths(args)
    
    method = args.method
    if not method == "connectivity_based_only":
        MolGraph.load_operations(paths.operation_path, args.num_operations)
    MolGraph.load_vocab(paths.vocab_path)

    make_trainig_data(
        mols_pkl_dir = paths.mols_pkl_dir,
        valid_path = paths.valid_path,
        vocab_path = paths.vocab_path,
        train_processed_dir = paths.train_processed_dir,
        valid_processed_dir = paths.valid_processed_dir,
        vocab_processed_path = paths.vocab_processed_path,
        num_workers = args.num_workers,
        methods = args.method,
    )

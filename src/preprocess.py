import os.path as path

import torch

from arguments import parse_arguments
from make_training_data import make_trainig_data
from merging_operation_learning import merging_operation_learning
from model.mol_graph import MolGraph
from model.mydataclass import Paths
from motif_vocab_construction import frequency_based_motif_vocab_construction, connectivity_based_motif_vocab_construction, vocab_ensemble, ensemble_mol_graphs, vocab_import,import_mol_graphs

if __name__ == "__main__":

    args = parse_arguments()
    paths = Paths(args)

    if not path.exists(paths.operation_path):
        learning_trace = merging_operation_learning(
            train_path = paths.train_path,
            operation_path = paths.operation_path,
            num_iters = args.num_iters,
            min_frequency = args.min_frequency,
            num_workers = args.num_workers,
            mp_threshold = args.mp_thd,
        )

    method = args.method
    if not method == "connectivity_based_only":
        MolGraph.load_operations(paths.operation_path, args.num_operations)
        
    if not path.exists(paths.vocab_path):
        if args.method == 'ensemble':
            frequency_based_motif_vocab_construction(
                train_path=paths.train_path,
                vocab_path=paths.freq_vocab_path,  # vocab constructed by frequency_based_motif_vocab_construction()
                operation_path=paths.operation_path,
                num_operations=args.num_operations,  # args.num_operations (default是500)
                mols_pkl_dir=paths.freq_mols_pkl_dir,
                # Train.smiles -> MolGraph(.pkl) set by frequency_based_motif_vocab_construction()
                num_workers=args.num_workers,  # args.num_workers (default是60)
            )
            connectivity_based_motif_vocab_construction(
                train_path=paths.train_path,
                vocab_path=paths.conn_vocab_path,  # vocab constructed by connectivity_based_motif_vocab_construction()
                mols_pkl_dir=paths.conn_mols_pkl_dir,
                # Train.smiles -> MolGraph(.pkl) set by connectivity_based_motif_vocab_construction()
                num_workers=args.num_workers,  # args.num_workers (default是60)
            )

            # 调用合并函数，将freq_vocab_path中的词典和conn_vocab_path中的词典合并，合并结果【引入工作区】——存储到paths.vocab_path
            vocab_ensemble(
                freq_vocab_path=paths.freq_vocab_path,
                conn_vocab_path=paths.freq_vocab_path,
                vocab_path=paths.vocab_path,
            )

            # 调用MolGraph数据的【集成函数】，将freq_mols_pkl_dir和conn_mols_pkl_dir中的文件集成后放入工作区
            ensemble_mol_graphs(
                freq_mols_pkl_dir=paths.freq_mols_pkl_dir,
                conn_mols_pkl_dir=paths.conn_mols_pkl_dir,
                mols_pkl_dir=paths.mols_pkl_dir,
                data_ensemble_mode=args.data_ensemble_mode,
            )

        elif args.method == 'frequency_based_only':
            # 直接将词典存储到paths.vocab_path
            frequency_based_motif_vocab_construction(
                train_path=paths.train_path,
                vocab_path=paths.freq_vocab_path,  # vocab constructed by frequency_based_motif_vocab_construction()
                operation_path=paths.operation_path,
                num_operations=args.num_operations,
                mols_pkl_dir=paths.freq_mols_pkl_dir,
                # Train.smiles -> MolGraph(.pkl) set by frequency_based_motif_vocab_construction()
                num_workers=args.num_workers,
            )

            # 调用【引入工作区】函数，将freq_vocab_path中的词典【引入工作区】——存储到paths.vocab_path
            vocab_import(
                target_vocab_path=paths.freq_vocab_path,
                source_vocab_path=paths.vocab_path,
            )
            # 调用MolGraph数据的【引入工作区函数】，将freq_vocab_path中的所有文件复制到工作区
            import_mol_graphs(
                source_mols_pkl_dir=paths.freq_mols_pkl_dir,
                target_mols_pkl_dir=paths.mols_pkl_dir,
            )

        elif args.method == 'connectivity_based_only':
            connectivity_based_motif_vocab_construction(
                train_path=paths.train_path,
                vocab_path=paths.conn_vocab_path,  # vocab constructed by connectivity_based_motif_vocab_construction()
                mols_pkl_dir=paths.conn_mols_pkl_dir,
                # Train.smiles -> MolGraph(.pkl) set by connectivity_based_motif_vocab_construction()
                num_workers=args.num_workers,
            )

            # 调用【引入工作区】函数，将conn_vocab_path中的词典【引入工作区】——存储到paths.vocab_path
            vocab_import(
                target_vocab_path=paths.conn_vocab_path,
                source_vocab_path=paths.vocab_path,
            )
            # 调用MolGraph数据的【引入工作区函数】，将conn_mols_pkl_dir中的所有文件复制到工作区
            import_mol_graphs(
                source_mols_pkl_dir=paths.conn_mols_pkl_dir,
                target_mols_pkl_dir=paths.mols_pkl_dir,
            )
    
    MolGraph.load_vocab(paths.vocab_path)
    
    torch.multiprocessing.set_sharing_strategy("file_system")
    make_trainig_data(
        mols_pkl_dir = paths.mols_pkl_dir,
        valid_path = paths.valid_path,
        vocab_path = paths.vocab_path,
        train_processed_dir = paths.train_processed_dir,
        valid_processed_dir = paths.valid_processed_dir,
        vocab_processed_path = paths.vocab_processed_path,
        num_workers = args.num_workers,
    )

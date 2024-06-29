import argparse
import logging
import os
import os.path as path
import random
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from arguments import parse_arguments
from model.dataset import MolsDataset, batch_collate
from model.MiCaM_VAE import MiCaM, VAE_Output
from model.mol_graph import MolGraph
from model.mydataclass import ModelParams, Paths, TrainingParams
from model.scheduler import beta_annealing_schedule


def train(args: argparse.Namespace):

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    paths = Paths(args)
    tb = SummaryWriter(log_dir=paths.tensorboard_dir)
    
    model_params = ModelParams(args)
    training_params = TrainingParams(args)

    method = args.method
    if not method == "connectivity_based_only":
        MolGraph.load_operations(paths.operation_path, args.num_operations)

    MolGraph.load_vocab(paths.vocab_path)

    os.makedirs(paths.output_dir)
    log_file = path.join(paths.output_dir, "train.log")
    print(f"See {log_file} for log." )
    logging.basicConfig(
        filename = log_file,
        filemode = "w",
        format = "[%(asctime)s]: %(message)s",
        level = logging.INFO
    )


    gc.collect()
    torch.cuda.empty_cache()

    model = MiCaM(model_params).cuda()
    optimizer = optim.Adam(model.parameters(), lr=training_params.lr)

    total_step, beta = 0, training_params.beta_min

    logging.info("HyperParameters:")
    logging.info(model_params)
    logging.info(training_params)

    scheduler = lr_scheduler.ExponentialLR(optimizer, training_params.lr_anneal_rate)
    beta_scheduler = beta_annealing_schedule(params=training_params, init_beta=beta, init_step=total_step)
    train_dataset = MolsDataset(paths.train_processed_dir)

    logging.info(f"Begin training...")
    os.makedirs(paths.model_save_dir)
    stop_train = False
    while True:
        for input in DataLoader(dataset=train_dataset, batch_size=training_params.batch_size, shuffle=True, collate_fn=batch_collate):
            total_step += 1
            model.zero_grad()

            input = input.cuda()
            output: VAE_Output = model(input, beta=beta, prop_weight=training_params.prop_weight)

            output.total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_params.grad_clip_norm)
            
            optimizer.step()
            output.log_tb_results(total_step, tb, beta, scheduler.get_last_lr()[0])

            if total_step % 10 == 0:
                output.print_results(total_step, lr=scheduler.get_last_lr()[0], beta=beta)

            if total_step % training_params.lr_anneal_iter == 0:
                scheduler.step()

            beta = beta_scheduler.step()

            if total_step == training_params.steps:
                stop_train = True
                break
        
        if stop_train: break
            
    model.eval()
    model.zero_grad()
    torch.cuda.empty_cache()
    model_path = path.join(paths.model_save_dir,"model.ckpt")
    motifs_embed_path = path.join(paths.model_save_dir,"motifs_embed.ckpt" )
    with torch.no_grad():
        ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
        torch.save(ckpt, model_path)
        model.save_motifs_embed(motifs_embed_path)
    
    logging.info(f"Benchmarking...")
    with torch.no_grad():
        model.load_state_dict(torch.load(model_path)[0])
        model.load_motifs_embed(motifs_embed_path)
        benchmark_results = model.benchmark(train_path=paths.train_path)
        logging.info(benchmark_results)
    tb.close()


# 将最后一次训练得到的模型进行benchmark测试，并生成对应的log
def test_on_benchmark(args: argparse.Namespace, choosed_output_dir: str):
    """
    将最后一次训练得到的模型进行benchmark测试，并生成对应的log
    """

    paths = Paths(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    method = args.method
    if not method == "connectivity_based_only":
        MolGraph.load_operations(paths.operation_path, args.num_operations)
    MolGraph.load_vocab(paths.vocab_path)

    
    # log设置
    log_file = path.join(choosed_output_dir, "benchmark_only.log")
    print(f"See {log_file} for log.")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="[%(asctime)s]: %(message)s",
        level=logging.INFO
    )


    # 模型定义与初始化
    model_params = ModelParams(args)  # 需要确保有args传入，或者从其他地方获取model_params
    model = MiCaM(model_params).cuda()


    # 路径设置
    model_path = path.join(choosed_output_dir, "ckpt", "model.ckpt")
    motifs_embed_path = path.join(choosed_output_dir, "ckpt", "motifs_embed.ckpt")



    # benchmark测试
    logging.info(f"Benchmarking...")
    with torch.no_grad():
        model.load_state_dict(torch.load(model_path)[0])
        model.load_motifs_embed(motifs_embed_path)
        benchmark_results = model.benchmark(train_path=paths.train_path)
        logging.info(benchmark_results)


if __name__ == "__main__":

    args = parse_arguments()
    torch.cuda.set_device(args.cuda)

    if args.benchmark_only == "0":
        train(args)
    elif args.benchmark_only == "1":
        # 调用test_on_benchmark
        if not args.choosed_output_dir:
            raise ValueError("choosed_output_dir must be specified for benchmarking_only.")
        test_on_benchmark(args, args.choosed_output_dir)    
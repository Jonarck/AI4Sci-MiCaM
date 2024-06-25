import argparse
import logging
import os
import os.path as path
import random

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

    # 超参数导入
    model_params = ModelParams(args)
    training_params = TrainingParams(args)

    MolGraph.load_operations(paths.operation_path)
    MolGraph.load_vocab(paths.vocab_path) # 这里需要注意

    # log设置
    os.makedirs(paths.output_dir)
    log_file = path.join(paths.output_dir, "train.log")
    print(f"See {log_file} for log." )
    logging.basicConfig(
        filename = log_file,
        filemode = "w",
        format = "[%(asctime)s]: %(message)s",
        level = logging.INFO
    )

    # 模型定义与初始化
    model = MiCaM(model_params).cuda()

    # 优化器初始化
    optimizer = optim.Adam(model.parameters(), lr=training_params.lr)

    # 超参数设置
    total_step, beta = 0, training_params.beta_min

    # log记录
    logging.info("HyperParameters:")
    logging.info(model_params)
    logging.info(training_params)

    # 训练
    scheduler = lr_scheduler.ExponentialLR(optimizer, training_params.lr_anneal_rate)
    beta_scheduler = beta_annealing_schedule(params=training_params, init_beta=beta, init_step=total_step)
    train_dataset = MolsDataset(paths.train_processed_dir) # 只导入了训练数据

    logging.info(f"Begin training...")
    os.makedirs(paths.model_save_dir)
    stop_train = False
    while True:
        for input in DataLoader(dataset=train_dataset, batch_size=training_params.batch_size, shuffle=True, collate_fn=batch_collate): # 【数据导入】发生
            total_step += 1
            model.zero_grad()

# collate_fn决定输入数据的形式
            input = input.cuda()

# model决定输入的转化：调用forward()函数
            output: VAE_Output = model(input, beta=beta, prop_weight=training_params.prop_weight)

# 反向传播
            output.total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_params.grad_clip_norm)
            
            optimizer.step()
            output.log_tb_results(total_step, tb, beta, scheduler.get_last_lr()[0])

            if total_step % 50 == 0:
                output.print_results(total_step, lr=scheduler.get_last_lr()[0], beta=beta)

            if total_step % training_params.lr_anneal_iter == 0:
                scheduler.step()

            beta = beta_scheduler.step()

            if total_step == training_params.steps:
                stop_train = True
                break
        
        if stop_train: break

# 保存结果与benchmark测试

    model.eval()
    model.zero_grad()
    torch.cuda.empty_cache()

    # 保存模型和motif的嵌入表示
    model_path = path.join(paths.model_save_dir,"model.ckpt")
    motifs_embed_path = path.join(paths.model_save_dir,"motifs_embed.ckpt" )
    with torch.no_grad():
        ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
        torch.save(ckpt, model_path)
        model.save_motifs_embed(motifs_embed_path)

    # bencmark测试
    logging.info(f"Benchmarking...")
    with torch.no_grad():
        model.load_state_dict(torch.load(model_path)[0])
        model.load_motifs_embed(motifs_embed_path)
        benchmark_results = model.benchmark(train_path=paths.train_path) # 在"train.smiles"上使用benchmark
        logging.info(benchmark_results) # 记录结果
    tb.close()

if __name__ == "__main__":

    args = parse_arguments()
    torch.cuda.set_device(args.cuda)

    train(args)
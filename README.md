# MiCaM: De Novo Molecular Generation via Connection-aware Motif Mining

This is the code of paper **De Novo Molecular Generation via Connection-aware Motif Mining**. *Zijie Geng, Shufang Xie, Yingce Xia, Lijun Wu, Tao Qin, Jie Wang, Yongdong Zhang, Feng Wu, Tie-Yan Liu.* ICLR 2023. [[arXiv](https://arxiv.org/pdf/2302.01129.pdf)]

## Environment
- Python 3.7
- Pytorch
- rdkit
- networkx
- torch-geometric
- guacamol
 
(**note:** FCD included with guacamol need to be FCD1.1, but `pip install guacamol` will install FCD 1.2, use `pip uninstall FCD` and `pip install FCD==1.1`, before benchmark)

## Workflow

Put the dataset under the `./data` directory. Name the training set and avlid set as `train.smiles` and `valid.smiles`, respectively. An example of the working directory is as following.
```
AI4Sci-MiCaM
├── data
│   └── guacamol
│       ├── train.smiles
│       └── valid.smiles
├── output/
├── preprocess/
├── src/
└── README.md
```

### 1. Mining connection-aware motifs

It consists of two phases: merging operation learning and motif vocabulary construction.

For merging operation learning, run the commands in form of

```
python src/merging_operation_learning.py \
    --dataset guacamol \
    --method ensemble  \
    --data_ensemble_mode random  \
    --num_workers 60
```
```
python src/merging_operation_learning.py \
    --dataset guacamol \
    --method ensemble  \
    --data_ensemble_mode overlay  \
    --num_workers 60
```
```
python src/merging_operation_learning.py \
    --dataset guacamol \
    --method frequency_based_only  \
    --num_workers 60
```
```
python src/merging_operation_learning.py \
    --dataset guacamol \
    --method connectivity_based_only  \
    --num_workers 60
```

For motif vocabulary constraction, run the commands in form of


### 2. Preprocess

To generate training data, using a given motif vocabulary, run the commands in form of

```
python src/make_training_data.py \
    --dataset guacamol \
    --num_operations 500 \
    --method ensemble  \
    --num_workers 60
```
```
python src/make_training_data.py \
    --dataset guacamol \
    --num_operations 500 \
    --method frequency_based_only  \
    --num_workers 60
```
```
python src/make_training_data.py \
    --dataset guacamol \
    --num_operations 500 \
    --method connectivity_based_only  \
    --num_workers 60
```


### 3. Training **MiCaM**

To train the MiCaM model, run a command in form of

```
python src/train.py \
    --benchmark_only 0 \
    --job_name train_micam \
    --dataset guacamol \
    --method ensemble \
    --num_operations 500 \
    --batch_size 2000 \
    --depth 15 \
    --motif_depth 6 \
    --latent_size 64 \
    --hidden_size 256 \
    --dropout 0.3 \
    --steps 30000 \
    --lr 0.005 \
    --lr_anneal_iter 50 \
    --lr_anneal_rate 0.99 \
    --beta_warmup 3000 \
    --beta_min 0.001 \
    --beta_max 0.3 \
    --beta_anneal_period 40000 \
    --prop_weight 0.2 \
    --cuda 0
```

Benchmarking will be automatically conduct during the training process.
You can also use Benchmark directly on a trained model, requiring you to provide the storage path to the model.ckpt file for that model, especially if Benchmark's Keras environment requirements are different from the training process


```
python src/train.py \
    --benchmark_only 1 \
    --choosed_output_dir [your model.ckpt path, e.g. output/06-29/00:59:59-train_micam/-QM9-frequency_based_only]
    --job_name train_micam \
    --dataset guacamol \
    --method ensemble \
    --num_operations 500 \
    --cuda 0
```





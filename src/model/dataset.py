import os
import os.path as path
from typing import List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from model.mol_graph import MolGraph
from model.mydataclass import batch_train_data, mol_train_data, train_data
from model.vocab import SubMotifVocab

# train_dataset = MolsDataset(paths.train_processed_dir)
class MolsDataset(Dataset):

    def __init__(self, data_dir: str) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.files = os.listdir(data_dir) # 【mol_train_data二进制pth文件列表】

    def __getitem__(self, index):
        file = path.join(self.data_dir, self.files[index]) # 从列表中定位【单个mol_train_data二进制pth文件】
        return torch.load(file)

    def __len__(self):
        return len(self.files) # 文件总量


# collate_fn——训练数据导入函数 DataLoader(dataset=train_dataset, batch_size=training_params.batch_size, shuffle=True, collate_fn=batch_collate):
def batch_collate(batch: List[mol_train_data]) -> batch_train_data:
    '''
    【MolGraph整体分子对应的data对象:  【x原子特征矩阵,edge_index边映射,edge_attr边类型矩阵】  】 ，
    【MolGraph的属性list:     [logP, Wt, qed, sa] 】，
    【MolGraph的起始Motif(最大)节点标签(Motif在Motif_vocab中的编号) 】 ，
    【分子内每一个query_atom连接位点的训练数据train_data对象构成的列表:
      →  [(
            【query_atom对应的连接过程子图】对应的data对象：【 x原子特征矩阵, edge_index边映射, edge_attr边类型矩阵】   ,
            【query_atom对应的连接位点在【该子图中】的序号】  ，
            【成环候选点列表】  ,
            【连接位置】
           )， ... ]
     】
    【MolGraph的Motif有序 (拼接顺序) 标签(Motif在Motif_vocab中的编号)列表】
    '''

    batch_mols_graphs: List[Data] = [] # Batch中用来接收【MolGraph整体分子对应的data对象】
    batch_props: List[torch.Tensor] = [] # Batch中用来接收【MolGraph的属性list】
    batch_start_labels: List[int] = [] # Batch中用来接收【MolGraph的起始Motif(最大)节点标签(Motif在Motif_vocab中的编号，简称为：motif的全局编号) 】
    batch_train_data_list: List[List[train_data]] = [] # Batch中用来接收【分子内每一个query_atom连接位点的训练数据data对象构成的列表】
    motif_lists: List[List[int]] = [] # Batch中用来接收【MolGraph的Motif有序 (拼接顺序) 标签(Motif在Motif_vocab中的编号)列表】

    for data in batch:
        batch_mols_graphs.append(data.mol_graph)
        batch_props.append(data.props)
        batch_start_labels.append(data.start_label) # data的【MolGraph的起始Motif(最大)节点标签(Motif在Motif_vocab中的编号) 】
        batch_train_data_list.append(data.train_data_list) # data的【分子内每一个query_atom连接位点对应的训练数据train_data对象构成的列表】
    # 【每一个query_atom连接位点对应的训练数据train_data对象】简称【一个“查询连接步骤的状态”的train_data】
        motif_lists.append(data.motif_list) # 将这个data的【MolGraph的Motif (按拼接顺序排列的)全局编号列表】放入batch的motif_lists列表

    # 【batch中所有的motif的全局编号】 motif_list中只包含了这个batch中出现了的motif全局编号
    motifs_list = list(set(sum(motif_lists, []))) # 将嵌套列表中的所有元素展平并去重，最后转换为一个新的列表，得到了【batch中出现的motif全局编号】
    # motif_lists中只包含了这个batch中出现了的motif全局编号，所以motif_vocab调用SubMotifVocab而非MotifVocab

'''
传入两个参数：1.【完整的motif_vocab对象】 2.【batch中出现的motif全局编号】
得到SubMotifVocab对象的实例，包含以下属性：
# motif_vocab：完整的motif词典对象实例 —— 【MOTIF_VOCAB】包含以下属性：
    # motif_vocab.motif_smiles_list : 【有连接点信息的motifs的smiles】列表 → （该列表的序号，就是【Motif在完整Motif_vocab中的编号(秩)】简称：【motif全局编号】）
    # motif_vocab.motif_vmap : motif【smiles-全局编号】映射词典：{key = "有连接点信息的motif的smiles"，value = "motif全局编号"}  给定motif有连接点的smiles，查找整数编号
    # motif_vocab.vocab_conn_dict : 【motif组织】连接位点的全局偏移量词典：以【motif全局编号】为键，以【单个motif的连接位点词典：{以【以连接位点在motif中的原子次序】为键，以【连接位点全局偏移量】为值} 】为值 —— 【给定(连接位点所在motif的编号，连接位点在motif中的原子次序)→【获取连接位点全局偏移量】
    # motif_vocab.conn_dict : 连接位点【全局定位】词典：以【连接位点全局偏移量】为键，以（【motif全局编号】，【连接位点在motif中的原子次序】）为值 —— 【连接位点全局偏移量】 → 【获取连接位点所在motif及在该motif中的原子次序】
    # motif_vocab.nodes_idx : 【连接位点的【原子偏移量】】列表：记录每个连接位点的【原子全局偏移量】
    # motif_vocab.num_atoms_dict : 【motif的规模】词典：以【motif全局编号】为键，以【motif的原子数量】为值
    # motif_vocab.bond_type_conns_dict : 【化学键组织】连接位点【列表】词典：以【化学键类型】为键，以【一种化学键对应的【所有连接位点全局偏移量构成的列表】】为值 —— 【给定【化学键】类型】→ 【获取该化学键对应的所有连接位点的【全局偏移量】】
# sublist: 【batch中出现的motif全局编号】列表
# idx2sublist_map: motif的【全局-batch】映射字典：{Motif的全局编号：Motif在sublist列表中的序号（简称为batch编号）}
# vocab_conn_dict: 【motif组织】连接位点的batch偏移量词典：以【motif全局编号】为键，以【单个motif的连接位点词典：{以【以连接位点在motif中的原子次序】为键，以【连接位点的batch偏移量】为值} 】为值 —— 【给定(连接位点所在motif的全局，连接位点在motif中的原子次序)→ 【获取连接位点的batch偏移量】
# nodes_idx: 连接位点的【原子batch偏移量】列表
# motif_idx_in_sublist_map: motif的【全局-batch】映射字典（等价于idx2sublist_map）：以【motif的全局编号】为键，【motif的batch编号】为值
'''
    motif_vocab = SubMotifVocab(MolGraph.MOTIF_VOCAB, motifs_list)

    motif_conns_idx = motif_vocab.get_conns_idx() # 获取——【全体】连接位点的【原子batch偏移量】列表
    motif_conns_num = len(motif_conns_idx) # 【全体】连接位点的数量

# batch_start_labels的【全局-batch映射】： 将【MolGraph的起始Motif(最大Motif)节点】的【motif的全局编号】转换为【motif的batch编号】
    batch_start_labels = [motif_vocab.motif_idx_in_sublist(idx) for idx in batch_start_labels]

# offset：每次为下次【查询连接位点】的【batch偏移量】的构建：增加本轮【“查询连接步骤”数据】的【原子特征矩阵的维度】==增加本轮【查询连接过程子图的原子数量】
# G_offset：【“查询连接步骤”数据】的【batch偏移量】
# conn_offset：【[0]号成环候选点】（在【成环类型】数据中，就是【成环目标点】）的【连接位点batch偏移量】
    offset, G_offset, conn_offset= 0, 0, motif_conns_num # 初始化为【全体连接位点的数量】，超出了【最大连接位点偏移量】，以表示【成环类型】
# batch_train_graphs列表——元素是：【“查询连接步骤”数据】的Data对象：【 x原子特征矩阵, edge_index边映射, edge_attr边类型矩阵】
    batch_train_graphs: List[Data] = []

# mol_idx列表——元素是：【“查询连接步骤”数据】【所在分子】的【分子数据batch编号】
# graph_idx列表——元素是：【“查询连接步骤”数据】的【batch偏移量】
# query_idx列表——元素是：【查询连接位点】的【原子batch偏移量】
# cyclize_cand_idx列表——元素是：【【查询连接位点】的所有【成环候选点】】的【原子batch偏移量】构成的【列表】
# labels列表——元素是：【目标连接位点】的【连接位点batch偏移量】
    # 成环类型的【label】: 连在【本motif（-1）】的【成环目标点——[0]号成环候选点】的【连接位点batch偏移量】
    # motif拼接类型的【label】：在另一个motif上的【拼接目标点】的【连接位点batch偏移量】
    mol_idx, graph_idx, query_idx, cyclize_cand_idx, labels = [], [], [], [], []

# 对【每一个分子】的训练数据（多个连接位点，对应多条【“查询连接步骤”数据】），进行真正的训练数据重整：
    for bid, data_list in enumerate(batch_train_data_list):
        # 遍历batch中的每一个分子：bid：【分子的训练数据】在batch中的序号；data_list：【分子的训练数据】
        for data in data_list: # 遍历一个分子内--【“查询连接步骤的状态”的train_data】构成的列表
            # 对于一个【“查询连接步骤的状态”的train_data】
            '''
            一个【“查询连接步骤的状态”的train_data】：
            (`graph `【该查询连接步骤对应的query_atom对应的连接过程状态子图】对应的data对象：【 x原子特征矩阵, edge_index边映射, edge_attr边类型矩阵】   ,
             `query_atom`【query_atom对应的连接位点在【该子图中】的序号】  ，
             `cyclize_cand`【成环候选点列表】  ,
             `label`【连接位置】)
            '''

            # 获取 【query_atom对应的连接位点在【该状态子图中】的序号】、【成环候选点列表】、【连接位置标签】
            query_atom, cyclize_cand, (motif_idx, conn_idx) = data.query_atom, data.cyclize_cand, data.label

# mol_idx列表——元素是：【“查询连接步骤”数据】【所在分子】的【分子数据batch编号】
            mol_idx.append(bid)
# graph_idx列表——元素是：【“查询连接步骤”数据】的【batch偏移量】
            graph_idx.append(G_offset)
# query_idx列表——元素是：【查询连接位点】的【原子batch偏移量】
            query_idx.append(query_atom + offset)
# cyclize_cand_idx列表——元素是：【【查询连接位点】的所有【成环候选点】】的【原子batch偏移量】构成的【列表】
            cyclize_cand_idx.extend([cand + offset for cand in cyclize_cand])
# labels列表——元素是：【目标连接位点】的【连接位点batch偏移量】
    # 成环类型的【label】: 连在【本motif（-1）】的【成环目标点——[0]号成环候选点】的【连接位点batch偏移量】
    # motif拼接类型的【label】：在另一个motif上的【拼接目标点】的【连接位点batch偏移量】
            if motif_idx == -1: # 成环类型的【label】: 连在【本motif（-1）】的【成环目标点——[0]号成环候选点】的【连接位点batch偏移量】
                labels.append(conn_offset)
            else: # motif拼接类型的【label】：在另一个motif上的【拼接目标点】的【连接位点batch偏移量】
                labels.append(motif_vocab.get_conn_label(motif_idx, conn_idx))# motif_vocab.get_conn_label是SubMotifVocab的get_conn_label：给定【motif的全局编号】，以及【以连接位点在motif中的原子次序】，获得【连接位点的batch偏移量】
# batch_train_graphs列表——元素是：【“查询连接步骤”数据】的Data对象：【 x原子特征矩阵, edge_index边映射, edge_attr边类型矩阵】
            batch_train_graphs.append(data.graph)
# offset：每次为下次【查询连接位点】的【batch偏移量】的构建：增加本轮【“查询连接步骤”数据】的【原子特征矩阵的维度】==增加本轮【查询连接过程子图的原子数量】
            offset += len(data.graph.x)
# G_offset：【“查询连接步骤”数据】的【batch偏移量】
            G_offset += 1
# conn_offset：【[0]号成环候选点】（在【成环类型】数据中，就是【成环目标点】）的【连接位点batch偏移量】
            conn_offset += len(cyclize_cand)

    return batch_train_data(
        batch_mols_graphs = Batch.from_data_list(batch_mols_graphs),
        batch_props = torch.Tensor(batch_props),
        batch_start_labels = torch.LongTensor(batch_start_labels),
        motifs_list = torch.LongTensor(motifs_list),
        batch_train_graphs = Batch.from_data_list(batch_train_graphs),
        mol_idx = torch.LongTensor(mol_idx),
        graph_idx = torch.LongTensor(graph_idx),
        query_idx = torch.LongTensor(query_idx),
        cyclize_cand_idx = torch.LongTensor(cyclize_cand_idx),
        motif_conns_idx = torch.LongTensor(motif_conns_idx),
        labels = torch.LongTensor(labels),
    )

'''
核心元素1. batch中全部分子的 【MolGraph【整体分子图】对应的data对象】 
batch_mols_graphs = Batch.from_data_list(batch_mols_graphs),
    1.【MolGraph【整体分子图】对应的data对象:【x原子特征矩阵,edge_index边映射,edge_attr边类型矩阵】 】
    2. Batch.from_data_list：将一组数据对象转换为一个批处理的Data对象，这样可以在批处理中高效地进行图神经网络的训练和推理。

核心元素2. batch中全部分子的【MolGraph的属性list】
batch_props = torch.Tensor(batch_props),
    1.【MolGraph的属性list: [logP, Wt, qed, sa] 】
    2. Tensor：将列表转换为张量，以便于在PyTorch中进行数值计算和梯度计算。

核心元素3. batch中全部分子的【MolGraph的起始Motif(最大)节点标签(Motif的全局编号) 】
batch_start_labels = torch.LongTensor(batch_start_labels),
    1.【MolGraph的起始Motif(最大)节点标签(Motif在Motif_vocab中的编号) 】
    2. LongTensor：将列表转换为长整型张量，以便在训练过程中进行索引和标签处理。

功能：【batch中出现的motif全局编号】
motifs_list = torch.LongTensor(motifs_list),
    1.【batch中出现的motif全局编号】列表
    2. LongTensor：将列表转换为长整型张量，以便在后续处理中作为索引使用。

核心元素4. 解包——batch中全部分子的【MolGraph分子内每一个query_atom连接位点的训练数据构成的列表】
batch_train_graphs = Batch.from_data_list(batch_train_graphs),
    1. batch_train_graphs列表——元素是：【“查询连接步骤”数据】的Data对象：【 x原子特征矩阵, edge_index边映射, edge_attr边类型矩阵】
    2. Batch.from_data_list：将一组数据对象转换为一个批处理的Data对象，这样可以在批处理中高效地进行图神经网络的训练和推理。

mol_idx = torch.LongTensor(mol_idx),
    1. mol_idx列表——元素是：【“查询连接步骤”数据】【所在分子】的【分子数据batch编号】
    2. LongTensor：将列表转换为长整型张量，以便在训练过程中对分子进行索引和分组。

graph_idx = torch.LongTensor(graph_idx),
    1. graph_idx列表——元素是：【“查询连接步骤”数据】的【batch偏移量】
    2. LongTensor：将列表转换为长整型张量，以便在训练过程中对图进行索引和批处理。

query_idx = torch.LongTensor(query_idx),
    1. query_idx列表——元素是：【查询连接位点】的【原子batch偏移量】
    2. LongTensor：将列表转换为长整型张量，以便在训练过程中对查询连接位点进行索引和定位。

cyclize_cand_idx = torch.LongTensor(cyclize_cand_idx),
    1. cyclize_cand_idx列表——元素是：【【查询连接位点】的所有【成环候选点】】的【原子batch偏移量】构成的【列表】
    2. LongTensor：将列表转换为长整型张量，以便在训练过程中对成环候选点进行索引和批处理。

motif_conns_idx = torch.LongTensor(motif_conns_idx),
    1. 获取——【全体】连接位点的【原子batch偏移量】列表
    2. LongTensor：将列表转换为长整型张量，以便在训练过程中对连接位点进行索引和批处理。

labels = torch.LongTensor(labels),
    1. labels列表——元素是：【目标连接位点】的【连接位点batch偏移量】
        1.1 成环类型的【label】: 连在【本motif（-1）】的【成环目标点——[0]号成环候选点】的【连接位点batch偏移量】
        1.2 motif拼接类型的【label】：在另一个motif上的【拼接目标点】的【连接位点batch偏移量】
    2. LongTensor：将列表转换为长整型张量，以便在训练过程中作为目标标签进行索引和批处理。
'''

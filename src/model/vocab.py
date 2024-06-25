import rdkit
import rdkit.Chem as Chem
from typing import List, Tuple, Dict
import torch
from model.utils import smiles2mol, get_conn_list
from collections import defaultdict


class load_vocab(object): # 基础的数据映射方法
    def __init__(self, vocab_list):
#
        self.vocab_list = vocab_list
        self.vmap = dict(zip(self.vocab_list, range(len(self.vocab_list))))
        
    def __getitem__(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab_list[idx]

    def size(self):
        return len(self.vocab_list)

class MotifVocab(object): #

    def __init__(self, pair_list: List[Tuple[str, str]]):

# 【有连接点信息的motifs的smiles】列表 → （该列表的序号，就是【Motif在完整Motif_vocab中的编号(秩)】简称：【motif全局编号】）
        self.motif_smiles_list = [motif for _, motif in pair_list] # 【完整的】Motif词汇对（无连接点信息的smiles,有连接点信息的smiles）列表 →  【有连接点信息的motifs的smiles】列表

# motif【smiles-全局编号】映射词典：{key = "有连接点信息的motif的smiles"，value = "motif全局编号"}  给定motif有连接点的smiles，查找整数编号
        self.motif_vmap = dict(zip(self.motif_smiles_list, range(len(self.motif_smiles_list))))


# 【所有种类原子】的全局偏移量：统计所有种类的原子在完整vocab【全局范围】的偏移量，简称【原子全局偏移量】
# 【连接位点】的全局偏移量：统计连接位点在完整vocab【全局范围】的偏移量，简称【连接位点全局偏移量】
# 【motif的规模】词典：以【motif全局编号】为键，以【motif的原子数量】为值
# 【连接位点的【原子偏移量】】列表：记录每个连接位点的【原子全局偏移量】
        node_offset, conn_offset, num_atoms_dict, nodes_idx = 0, 0, {}, []

# 【motif组织】连接位点的全局偏移量词典：以【motif全局编号】为键，以【单个motif的连接位点词典：{以【以连接位点在motif中的原子次序】为键，以【连接位点全局偏移量】为值} 】为值 —— 【给定(连接位点所在motif的编号，连接位点在motif中的原子次序)→ 【获取连接位点全局偏移量】
        vocab_conn_dict: Dict[int, Dict[int, int]] = {}
# 连接位点【全局定位】词典：以【连接位点全局偏移量】为键，以（【连接位点所在的Motif在完整Motif_vocab中的编号】，【连接位点在motif中的原子次序】）为值 —— 【给定连接位点在完整Motif_vocab的全局偏移量 → 获取连接位点所在motif及在该motif中的原子次序】
        conn_dict: Dict[int, Tuple[int, int]] = {}
# 【化学键组织】连接位点【列表】词典：以【化学键类型】为键，以【一种化学键对应的【所有连接位点全局偏移量构成的列表】】为值 —— 【给定【化学键】类型】→ 【获取该化学键对应的所有连接位点的【全局偏移量】】
        bond_type_motifs_dict = defaultdict(list)

        for motif_idx, motif_smiles in enumerate(self.motif_smiles_list):
            #
            motif = smiles2mol(motif_smiles)
            # 将motif中的所有原子节点按照【原子等级】排序
            ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=False)) # 不breakTies

            cur_orders = []
            # 构建【完整motif_vocab的vocab_conn_dict】→ 以【motif全局编号】为键，以【单个motif的连接位点词典】为值
            vocab_conn_dict[motif_idx] = {} # 【单个motif的连接位点词典】：以【以连接位点在motif中的原子次序】为键，以【连接位点全局偏移量】为值
            for atom in motif.GetAtoms():
                if atom.GetSymbol() == '*' and ranks[atom.GetIdx()] not in cur_orders:
                    # 如果遍历到的原子是连接位点，且原子未遍历过（not in cur_orders确保每个连接位点只被处理一次），则更新【该motif的连接位点词典】
                    bond_type = atom.GetBonds()[0].GetBondType()
                    # 更新【【motif组织】连接位点的全局偏移量词典】的操作：以【连接位点在motif中的原子次序】为键，以【连接位点全局偏移量】为值
                    vocab_conn_dict[motif_idx][ranks[atom.GetIdx()]] = conn_offset
                    # 更新【连接位点全局定位词典】：以【连接位点全局偏移量】为键，以（【连接位点所在的Motif的全局】，【连接位点在motif中的原子次序】）为值
                    conn_dict[conn_offset] = (motif_idx, ranks[atom.GetIdx()])

                    cur_orders.append(ranks[atom.GetIdx()])# 用于确保每个连接位点只被处理一次。
                    # 更新bond_type_motifs_dict
                    bond_type_motifs_dict[bond_type].append(conn_offset)

                    # 获取每个【连接位点的【原子全局偏移量】】，构成列表
                    nodes_idx.append(node_offset)

                    conn_offset += 1 #用于获取【连接位点】的全局偏移量
                node_offset += 1 #用于获取【所有种类原子】的全局偏移量
            # 更新【motif的规模】词典：以【motif全局编号】为键，以【motif的原子数量】为值
            num_atoms_dict[motif_idx] = motif.GetNumAtoms()




# 【motif组织】连接位点的全局偏移量词典：以【motif全局编号】为键，以【单个motif的连接位点词典：{以【以连接位点在motif中的原子次序】为键，以【连接位点全局偏移量】为值} 】为值 —— 【给定(连接位点所在motif的编号，连接位点在motif中的原子次序)→【获取连接位点全局偏移量】
        self.vocab_conn_dict = vocab_conn_dict
# 连接位点【全局定位】词典：以【连接位点全局偏移量】为键，以（【motif全局编号】，【连接位点在motif中的原子次序】）为值 —— 【连接位点全局偏移量】 → 【获取连接位点所在motif及在该motif中的原子次序】
        self.conn_dict = conn_dict
# 【连接位点的【原子偏移量】】列表：记录每个连接位点的【原子全局偏移量】
        self.nodes_idx = nodes_idx
# 【motif的规模】词典：以【motif全局编号】为键，以【motif的原子数量】为值
        self.num_atoms_dict = num_atoms_dict
# 【化学键组织】连接位点【列表】词典：以【化学键类型】为键，以【一种化学键对应的【所有连接位点全局偏移量构成的列表】】为值 —— 【给定【化学键】类型】→ 【获取该化学键对应的所有连接位点的【全局偏移量】】
        self.bond_type_conns_dict = bond_type_motifs_dict


    def __getitem__(self, smiles: str) -> int: # 输入smiles返回motif全局编号
        if smiles not in self.motif_vmap:
            print(f"{smiles} is <UNK>")
        return self.motif_vmap[smiles] if smiles in self.motif_vmap else -1

    # 【给定(连接位点所在motif的编号，连接位点在motif中的原子次序)→获取【连接位点全局偏移量】
    def get_conn_label(self, motif_idx: int, order_idx: int) -> int:
        return self.vocab_conn_dict[motif_idx][order_idx]

    # 获取【连接位点的【原子偏移量】】列表
    def get_conns_idx(self) -> List[int]:
        return self.nodes_idx
    # 给定【连接位点全局偏移量】 → 【获取连接位点所在motif及在该motif中的原子次序】
    def from_conn_idx(self, conn_idx: int) -> Tuple[int, int]:
        return self.conn_dict[conn_idx]

class SubMotifVocab(object):

    def __init__(self, motif_vocab: MotifVocab, sublist: List[int]):
        '''
        # 1. 完整的MOTIF_VOCAB对象（含有各种属性）
        def load_vocab(cls, vocab_path: str):
            pair_list = [line.strip("\r\n").split() for line in open(vocab_path)] # Motif词汇对（无连接点信息的smiles,有连接点信息的smiles）列表
            MolGraph.MOTIF_VOCAB = MotifVocab(pair_list) # 完整的motif_vocab
            MolGraph.MOTIF_LIST = MolGraph.MOTIF_VOCAB.motif_smiles_list # self.motif_smiles_list = [motif for _, motif in pair_list] 单纯的【有连接点信息的motif的smiles】列表

        # 2. 【batch中出现的motif全局编号】
        for data in batch:
            motif_lists.append(data.motif_list)# 将这个data的【MolGraph的Motif (按拼接顺序排列的)全局编号列表】放入batch的motif_lists列表
        motifs_list = list(set(sum(motif_lists, []))) # 将嵌套列表中的所有元素展平并去重，最后转换为一个新的列表，得到了【batch中出现的motif全局编号】

        motif_vocab = SubMotifVocab(MolGraph.MOTIF_VOCAB, motifs_list)
        '''
# 完整的motif词典 —— 【MOTIF_VOCAB】
        self.motif_vocab = motif_vocab
# 【batch中出现的motif全局编号】列表
        self.sublist = sublist
# motif的【全局-batch】映射字典：{Motif的全局编号：Motif在sublist列表中的序号（简称为batch编号）}
        self.idx2sublist_map = dict(zip(sublist, range(len(sublist))))

# 原子的batch偏移量
# 连接位点的batch偏移量
# 连接位点的【原子batch偏移量】列表
        node_offset, conn_offset, nodes_idx = 0, 0, []

# motif的【全局-batch】映射字典（等价于idx2sublist_map）：以【motif的全局编号】为键，【motif的batch编号】为值
        motif_idx_in_sublist = {}

# 【motif组织】连接位点的batch偏移量词典：以【motif全局编号】为键，以【单个motif的连接位点词典：{以【以连接位点在motif中的原子次序】为键，以【连接位点的batch偏移量】为值} 】为值 —— 【给定(连接位点所在motif的全局，连接位点在motif中的原子次序)→ 【获取连接位点的batch偏移量】
        vocab_conn_dict: Dict[int, Dict[int, int]] = {}

        # 遍历sublist：【i:motif的batch编号】 【mid：motif的全局编号】
        for i, mid in enumerate(sublist):
            motif_idx_in_sublist[mid] = i # 以【motif的全局编号】为键，【motif的batch编号】为值

            # 以【motif全局编号】为键，以【{cid:conn_offset}词典】为值，构建词典
            vocab_conn_dict[mid] = {}# mid：连接位点所在的motif全局编号
            for cid in motif_vocab.vocab_conn_dict[mid].keys():
                # motif_vocab.vocab_conn_dict ——【motif组织】连接位点词典：以【motif全局编号】为键，以【单个motif的连接位点词典：{以【以连接位点在motif中的原子次序】为键，以【连接位点全局偏移量】为值} 】为值
                # → cid: 连接位点在motif中的原子次序
                # → 按连接位点的次序，遍历这个motif中所有的连接位点
                vocab_conn_dict[mid][cid] = conn_offset  # 【{cid:conn_offset}词典】：以【连接位点在motif中的原子次序】为键，以【连接位点的batch偏移量】为值

                # 连接位点的【原子batch偏移量】列表
                nodes_idx.append(node_offset + cid) # 连接位点的【原子batch偏移量】
                conn_offset += 1 # 连接位点的batch偏移量
            node_offset += motif_vocab.num_atoms_dict[mid] # 下一个motif的【起始原子batch偏移量】：当前连接位点所在的motif的原子数量

# 【motif组织】连接位点的batch偏移量词典：以【motif全局编号】为键，以【单个motif的连接位点词典：{以【以连接位点在motif中的原子次序】为键，以【连接位点的batch偏移量】为值} 】为值 —— 【给定(连接位点所在motif的全局，连接位点在motif中的原子次序)→ 【获取连接位点的batch偏移量】
        self.vocab_conn_dict = vocab_conn_dict
# 连接位点的【原子batch偏移量】列表
        self.nodes_idx = nodes_idx
# motif的【全局-batch】映射字典（等价于idx2sublist_map）：以【motif的全局编号】为键，【motif的batch编号】为值
        self.motif_idx_in_sublist_map = motif_idx_in_sublist
    
    def motif_idx_in_sublist(self, motif_idx: int):
        return self.motif_idx_in_sublist_map[motif_idx]  # 给定【motif的全局编号】，获得【motif的batch编号】

    def get_conn_label(self, motif_idx: int, order_idx: int):
        return self.vocab_conn_dict[motif_idx][order_idx] # 给定【motif的全局编号】，以及【以连接位点在motif中的原子次序】，获得【连接位点的batch偏移量】
    
    def get_conns_idx(self):
        return self.nodes_idx # 连接位点的【原子batch偏移量】列表

    



